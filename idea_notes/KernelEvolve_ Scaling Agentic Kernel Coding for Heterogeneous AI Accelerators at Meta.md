## KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

- baseline方法是什么？
  Baseline是**手动kernel开发**（manual kernel development）——由领域专家为每个算子-硬件平台组合手写优化kernel。在Meta的生产环境中，面临三维度的诅咒（Curse of Dimensionality）：模型架构多样性（≥1500个模型，从MLP到Transformer-based ranking）、kernel原语多样性（200+ data preprocessing operators + compute kernels）、硬件代际和架构异构性（NVIDIA Ampere/Hopper, AMD CDNA3/CDNA4, MTIA v2i/v3）。这种组合爆炸导致O(operators × hardware platforms)的实现矩阵，每个平台特定实现需要2-8周的专家优化工作，且12-18个月的硬件更新周期使已有优化失效。

  此外，baseline还包括现有的AI-powered kernel coding research prototypes（AutoTriton, KernelLLM, GEAK-agent, Kevin, KernelAgent, TritorX, AlphaEvolve），它们存在六个根本缺陷：(1) 窄优化范围——target isolated subproblems，无end-to-end lifecycle management；(2) synthetic evaluation——canonical operators with static shapes，非production dynamic batching/variable sequence lengths/domain-specific transformations；(3) 单平台focus——homogeneous NVIDIA environments；(4) limited agent capabilities——缺乏fully autonomous workflows（multi-level verification + hierarchical profiling + persistent knowledge bases）；(5) 无inference-time scaling——无大规模搜索策略（greedy/MCTS/evolutionary）；(6) 无checkpointing——失败后从零开始。

  全栈执行例子（以NVIDIA H100上conv1d为例，手动开发流程）：
  - 算法层：Convolutional Transformer模型需要1D convolution over user event sequences（production shape: B=2048, Cin=96, Cout=96, L=200）。
  - 系统框架层：PyTorch torch.nn.functional.conv1d直接执行，内部调用cuDNN implicit GEMM，但需要NCHW↔NHWC layout转换（5次kernel launch含多次format conversion）。或者用conv2d workaround——reshape到4D + channels_last → cuDNN NHWC optimized path（4次kernel launch）。
  - 编译框架层：cuDNN/Triton compiler执行固定的compilation passes——静态tiling heuristics、generic autotuning templates——无awareness of production shape distribution。
  - kernel调度层：多个独立kernel launches（layout transform + GEMM + post-processing），每个kernel独立从HBM读取输入、写入输出，中间tensor通过HBM传递（redundant global memory traffic）。
  - 硬件架构层：NVIDIA H100 TMA和warp specialization能力未被利用——kernel使用传统warp-centric模式，无async copy、无double-buffering、无differentiated cache modifiers。

  Baseline两大核心缺陷：
  (1) **Preprocessing kernel缺失导致disaggregated serving architecture**：当关键preprocessing operators缺少native accelerator实现时，生产系统被迫采用disaggregated topology——preprocessing在CPU server执行，neural network在accelerator执行——引入10-20ms pure network overhead（P99 latency从61ms增至97ms，25% degradation），消耗sub-100ms latency budget的显著部分。这不是增量性能损失，而是binary deployment constraint——单个缺失operator block整个模型在accelerator上的monolithic deployment。
  (2) **手动开发无法scale到组合爆炸空间**：O(operators × hardware platforms × model stages)的实现矩阵，每个平台2-8周专家开发时间，新硬件12-18个月更新周期导致已有优化失效。手动开发的经济成本和组织负担使完整kernel coverage成为不可能——直接限制模型创新和部署速度。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出KernelEvolve——一个将kernel优化建模为**graph-based search with LLM agents**的自动化框架，通过四个关键设计解决baseline缺陷：

  **(1) 图搜索驱动的自动化kernel优化（Graph-Based Search & State Machine）**
  将kernel优化formalize为图搜索问题 G_t = (V_t, E_t)，specified by (F, π_sel, O, τ)。Fitness function F(v) = t_pytorch / t_triton；Selection policy π_sel支持greedy/MCTS/evolutionary三种策略；Universal Operator O 统一所有优化操作（Draft/Debug/Improve）为单个context-aware transformation；Termination rule τ基于时间/max artifacts/stall/fitness threshold。这替代了手动trial-and-error开发过程——将"人工数周探索"转化为"agent数小时搜索"（如conv1d 300 steps搜索自动发现fusion + tiling + autotuning + 3D grid + double-buffering + cache modifiers组合）。

  **(2) Universal Operator + Retrieval-Augmented Dynamic Prompting（Section 3.2）**
  用**单个通用算子**替代传统多算子框架（Draft/Debug/Improve各自有固定静态prompt template）。通过retrieval-augmented dynamic prompting机制，在每次迭代时基于实际runtime context（profiling结果、错误诊断、硬件约束）动态合成prompt——而非通过预定义的operator lens限制LLM推理。具体架构：
  - Context Memory Sub-Agent：分析runtime artifacts（kernel源码、profiling metrics、error diagnostics）→ 诊断bottleneck（如30% occupancy + high shared memory pressure → register spilling root cause）→ 生成优化指令
  - Deep Search Sub-Agent：根据诊断结果参数化知识库检索 → 检索目标hardware/optimization文档 → 注入LLM context window
  - Dynamic Prompt Synthesis：组合当前kernel + analysis report + retrieved knowledge + hardware constraints → 在token budget (64K-1M)内维持task-relevant information
  - Persistent Knowledge Base：hierarchical file system组织 → constraints/（correctness） + guidance/（platform-agnostic optimization） + hardware/{nvidia|amd|mtia}/（≥100 documents per platform）→ 通过index.md实现structured navigation

  **(3) MTIA Knowledge Injection for Proprietary Accelerators（Section 3.2.3）**
  针对MTIA（proprietary architecture absent from LLM pretraining corpora）的特殊挑战：系统化地将MTIA domain expertise注入持久化知识库。包括：libdevice API映射（SFU LUT operations如exp/gelu/sigmoid映射到专用硬件指令而非数学近似）、hardware-specific compilation options（cb_multiplier扩大circular buffer、use_dual_core分配DMA到core A+vector到core B实现pipeline parallelism）、compute helper functions（unary_elemwise_compute/binary_elemwise_compute/binary_elemwise_const_compute mapping到optimized vector instructions）、custom type system（TensorView/CoreID/ExecutionGrid via @core.struct_type decorator）、advanced synchronization（cross-PE broadcasting via tl.load direction attribute、cross-PE reduction via tl.store、runtime barriers via tl.pe_runtime_barrier、explicit tensor copies via tl.copy）。当LLM收到MTIA-targeted query时，retrieved documentation effectively teaches the model MTIA-specific idioms absent from pretraining——从"generate GPU-targeted Triton code that fails on MTIA"转变为"generate production-grade MTIA kernels leveraging hardware-specific features"。

  **(4) Multi-Granularity Evaluation & Profiling Integration（Section 3.4）**
  建立完整的多层次evaluation pipeline：TritonBench（correctness验证 + speedup测量）→ Torch Profiler（system-level timeline: CPU/GPU time, launch overhead）→ NCU（kernel-level: occupancy, memory throughput, instruction mix）→ Triton Proton/Triton MPP（intra-kernel: instruction-level pipeline behavior, async overlap）→ MTIA Insight（MTIA-specific: PE utilization, SFU/DPE/MLU utilization, cache hit rates, per-PE counters）。关键创新：Triton MPP作为unified profiling abstraction——通过compiler-centric job graph组合instrumentation、profiling passes、trace synthesis——解决"性能信号分散在DSL/compiler IR/CUDA/PTX/SASS/runtime/hardware counters多个abstraction layer"的fragmentation问题。Evaluation Code Generator（deterministic）自动将kernel artifact转换为instrumented evaluation scripts，通过FaaS platform dispatch到remote hardware worker——消除generation (CPU-bound) 和 evaluation (accelerator-bound) 之间的resource contention。

  **(5) Persistent Storage + Checkpointing for Scalable Search（Section 3.2.2）**
  Metadata store（关系数据库） + Object store（kernel文件）的两层存储架构支持：(a) distributed concurrent exploration——多个agent同时扩展不同节点，transaction isolation保证consistency；(b) complex contextual queries——通过SQL/recursive CTE实现graph traversal（如查找sibling outcomes、ancestor strategies、global best）；(c) cross-session knowledge reuse——历史optimized kernels作为新搜索的初始化（如在AMD MI350上类似GEMM变体：识别15个历史GEMM kernel、找到3个>1.5×speedup版本、以此为基础开始search）；(d) fault tolerance and checkpointing——persist每步搜索状态，crash后从last successful iteration恢复。这解决了"multi-hour optimization runs brittle and resource-inefficient"问题。

  全栈执行对比baseline（以NVIDIA H100上conv1d为例，KernelEvolve自动化流程）：
  - 算法层：同一conv1d计算，KernelEvolve在300步搜索中自动发现最优tiling和fusion策略组合。
  - 系统框架层：不再调用PyTorch conv1d/conv2d → cuDNN路径，而是使用KernelEvolve-generated fused Triton kernel，通过TritonBench BenchmarkOperator wrapper集成到模型inference pipeline。
  - 编译框架层：KernelEvolve的Universal Operator替代了固定compilation passes——通过retrieval-augmented prompting在每个search step动态调整优化策略，而非应用static tiling heuristics。Triton compiler接收已优化的Triton源码进行final compilation。
  - kernel调度层：从5次独立kernel launch（layout transform × 2 + GEMM + layout transform + post-process）→ 2次kernel launch（pack_conv1d_weight_kernel + conv1d_gemm_kernel）。跨operation fusion消除冗余layout转换和intermediate tensor materialization。使用3D grid launch并行化grouped convolution channels。使用double-buffered execution overlap memory access with Tensor Core operations。使用differentiated cache modifiers（.ca for streaming activations, .cg for reused weights）。
  - 硬件架构层：20+ autotune configurations探索BLOCK_M/N/K + num_warps + num_stages + WARP_SPECIALIZE组合空间。知识库驱动检索NVIDIA H100-specific文档（tensor cores + TMA + warp specialization + Hopper pipeline patterns）。Fitness score从~2000收敛到6889（300 steps），最终2.30× speedup vs PyTorch conv1d, 1.62× vs conv2d workaround。

  跨平台扩展（MTIA v3 conv1d）：
  - KernelEvolve的同一operator specification自动生成MTIA-specific kernel，知识库注入使LLM学习MTIA-specific idioms（libdevice API for SFU activation、cross-PE broadcasting for multi-PE kernels、cb_multiplier/use_dual_core for pipeline optimization），达到6.54× speedup vs PyTorch conv1d baseline——证明automated synthesis在vendor library coverage不成熟的custom accelerators上价值最大。
  - MapIdTransform在MTIA v2i上实现3.28-4.07× speedup——不仅优化性能，更是**唯一可行的on-device执行路径**（PyTorch baseline因缺少native ATen ops需CPU fallback）。实现从"missing kernel → disaggregated serving → 10-20ms network overhead"到"generated kernel → monolithic accelerator deployment → zero network tax"的架构转变。
