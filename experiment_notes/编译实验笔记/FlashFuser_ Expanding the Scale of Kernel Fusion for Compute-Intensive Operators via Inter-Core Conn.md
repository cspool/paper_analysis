## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- 属于编译框架的实现是什么？实验比较什么？
  FlashFuser 是一个基于 NVIDIA CUTLASS 的代码生成编译框架，利用 H100 GPU 的 Distributed Shared Memory（DSM，即 inter-core connection）扩展 kernel fusion 的可融合算子规模。核心实现包括三部分：(1) dsm_comm primitive——形式化 cluster 级别数据交换模式（shuffle、reduce、multiply），为 DSM-based fusion plan 提供统一表示；(2) Dataflow Analyzer——将 loop scheduling、tile selection、resource mapping 推广到 reg→SMEM→DSM 三级存储层次，通过贪心 spill 策略量化跨层数据搬移量；(3) Fusion Search Engine——使用解析 cost model（minimax formulation: min max C_l = V_l/B_l）和 5 条 pruning 规则（Divisible Tile Sizes, Cluster Size Constraint, Activation Constraint, Dependency Constraint, Memory Capacity Limit）从约 2.75×10^13 的搜索空间中高效探索最优 execution plan。前端为 Python-based search engine，后端为基于 CUTLASS 的 CUDA code generator。搜索离线进行，运行时通过 binning+table lookup 适应动态变化的 M 维度。
  实验比较：(1) GEMM chains（10 种配置 from DLRM/GPT/OPT/BERT/Performer）vs PyTorch+torch.compile、TensorRT、TVM/Relay、TASO、BOLT、Chimera；(2) Convolution chains（8 种配置 from ResNet）vs 同样 baselines；(3) Gated FFNs（8 种配置 from Llama/Qwen 系列）vs 同样 baselines；(4) 端到端推理（基于 SGLang）vs SGLang default；(5) Ablation study——全系统 vs DC+DA vs DA only；(6) Cost model validation 和 Top-K 分析；(7) dsm_comm primitive bandwidth/utilization 测试；(8) 搜索时间 vs Brute-Force；(9) 大模型（Llama3-70B, Qwen2.5-14B/32B）端到端 speedup。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU (SXM)，双路 Intel Xeon Platinum 8468 CPU (96 cores, 2.10GHz)。软件栈：CUDA 12.4, PyTorch 2.6, TVM 0.9, Triton 3.2, Nsight Compute 2025.2.0。端到端评估基于 SGLang。

- 开源编译框架是什么。修改了什么。
  基于 CUTLASS（https://github.com/NVIDIA/cutlass）的代码生成框架。修改包括：(1) 前端 search engine——Python 实现，枚举 LoopSchedule、TilingSize、ResourceMapping 组合，调用 Dataflow Analyzer 量化数据搬移，使用 cost model + 5 条 pruning 规则过滤候选；(2) 后端 code generator——扩展 CUTLASS kernel 结构（prologue-mainloop-epilogue），在 prologue 中初始化 DSM semaphore，在 mainloop 中注入 dsm_comm 操作（dsm_all_exchange, dsm_shuffle, dsm_reduce_scatter），在 epilogue 中执行 hierarchical reduction；(3) dsm_comm 实现——基于 TMA（Tensor Memory Accelerator）进行数据搬移，使用 mbarrier intrinsic 实现 many-to-many 同步（区别于 CUTLASS 原生的 all-to-one cluster-sync），支持 ring communication for SHUFFLE；(4) inter-cluster reduction 使用 TMA cp.reduce.async.bulk 指令进行跨 cluster 原子归约。论文未明确声明独立开源仓库（2025年12月 arXiv）。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  论文未提供独立开源仓库链接。FlashFuser 基于开源 CUTLASS 构建，编译框架使用流程：

  1. **输入**：高层 DNN model description（DNN graph g），device information d（H100 的 memory hierarchy: reg capacity, SMEM capacity 227KB/SM, DSM bandwidth per cluster size, L2/global bandwidth），Top-K count k=11。输入 graph 描述 GEMM chain 的维度 (M, N, K, L) 或 Gated FFN 的 branch 结构。
  2. **搜索空间枚举（Search Engine）**：
     - Loop Schedule：4 个独立维度 {M,N,K,L} 划分为 Spatial (parallel across SMs) 和 Temporal (sequential within SM)，共 41 种组合
     - Tile Size：cluster-level tile（5^4 种 cluster config，每个维度选自 {1,2,4,8,16}）+ block-level tile（以 MMA tile 16×16×16 为最小单位，从 problem size 除以 MMA tile 的倍数中选择）
     - Resource Mapping：默认 DSM 为 lowest-level cache
     - 初始搜索空间约 2.75×10^13 种可能
  3. **Pruning（5 条规则顺序应用）**：
     - Rule 1 (Divisible Tile): tile size 需整除 problem size 维度（from MC-Fuser [55]）
     - Rule 2 (Cluster Size): 每 GEMM 的 cluster dims product ≤ 16（H100 hardware limit），连续 GEMM 的 cluster dims 必须相同
     - Rule 3 (Activation): 前序 GEMM 的 accumulation dim 必须在最内层循环（保证 partial sum 完整可用于 activation）
     - Rule 4 (Dependency): L 维度不能设为 spatial（否则不同 spatial tile 的中间 C 无法直接通信）
     - Rule 5 (Memory Capacity): tensor 不能超过其可 spill 的最低级 cache 容量
     - Pruning 后约 1.15×10^6 个候选（GPT-6.7B），总缩减率 >99.99%
  4. **Dataflow Analyzer 评估（Algorithm 1）**：对每个 pruned candidate (s, t, r)：
     - 对每个 tensor 计算 Data Footprint (DF)
     - Input/Output tensors：沿 reversed(s) 迭代相关维度，计算 global memory data movement volume
     - Reused tensors：贪心从 reg→SMEM→DSM 逐级放置，计算每级 data movement volume
     - 输出 D_V (total data movement volume across all levels) 和 final plan p_final
  5. **Cost Model 排序**：对每个 candidate，计算 bottleneck cost C = max_l (V_l / B_l)，选择 min max C 的 Top-K candidates（K=11 时 accuracy≈100% of optimal）
  6. **硬件 Profiling**：Top-K candidates 经后端生成 CUDA kernel → 在 H100 上实测 → 选最优
  7. **代码生成（Back-End）**：
     - Prologue：初始化 DSM semaphore (mbarrier)
     - Mainloop：注入 dsm_comm 操作——GEMM0 后 dsm_all_exchange（AllReduce accumulation）→ GEMM1 中 dsm_shuffle（ring communication 交换 C tile）→ Store phase dsm_scatter_reduce（hierarchical intra-cluster + inter-cluster reduction）
     - Epilogue：通过 TMA cp.reduce.async.bulk 执行跨 cluster atomic reduction，存储最终 output
     - 生成 CUDA code 嵌入 CUTLASS kernel 模板
  8. **运行时 kernel 选择**：由于 FFN/conv 场景中只有 M 维度动态变化（N,K,L 固定），搜索离线完成，运行时通过 binning + table lookup 根据 M 选择预编译 kernel
  9. **输出**：高性能 fused CUDA kernel，在 GEMM chains 上平均 3.1× over PyTorch，4.1× over SOTA compilers (Chimera)；端到端 SGLang 上 1.24× speedup；全局显存访问减少 58%。

- 属于编译框架的实现是什么？实验比较什么？
  AccelOpt 是一个自改进 LLM agentic 系统，通过 Planner-Executor-Summarizer 三代理工作流 + beam search + optimization memory，自动探索 NKI kernel 优化空间，在不依赖人工优化知识的前提下将 baseline kernel 迭代优化为更高效的 kernel。实验比较：(1) AccelOpt vs Claude Sonnet 4 重复采样，(2) beam search vs 重复采样，(3) beam search + optimization memory vs beam search only，(4) Reflexion-style baseline，(5) 不同 executor/planner 模型和 memory 配置的 cost-benefit 分析。

- 硬件平台是什么，配置是什么。
  Amazon Trainium 1 (trn1.32xlarge EC2) 和 Trainium 2 (trn2.48xlarge EC2)。Trainium 1 单核: PeakBW 440.2 GB/s, PeakMM 23.75 TFLOPS, PeakVec 286.8 GFLOPS。Trainium 2 单核: PeakBW 640.0 GB/s, PeakMM 19.75 TFLOPS, PeakVec 550.0 GFLOPS。

- 开源编译框架是什么。修改了什么。
  Neuron Compiler（AWS 官方编译器）将 NKI kernel 编译为 Trainium 可执行代码。AccelOpt 不修改 Neuron Compiler 本身，而是通过 agentic workflow 自动生成优化的 NKI kernel 源码。Agent 系统使用 Neuron Profile 获取性能反馈，以 Roofline 模型计算 peak throughput percentage 作为评估指标。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源地址: https://github.com/zhang677/AccelOpt
  使用流程:
  1. 输入：baseline NKI kernel（由 Neuron Compiler 生成或人工编写的 NKI 源码）+ operator 问题描述 + profiling 信息
  2. Planner agent：分析 profile 识别性能瓶颈（如低 HFU、高 memory write），提出 1-step 优化计划（如 "Hoist LHS Transpose Out of Reduction Loop"）。每轮每个 candidate kernel 生成 N 个 plan。
  3. Executor agent：将优化计划转化为可执行的 NKI kernel 代码（每个 plan 尝试 K 次），应用 loop transformation、tiling、memory layout 变更等。
  4. Profiling Service：分布式 profiling 服务在 Trainium 硬件上运行生成的 kernel，测量 latency、HBM 读写、engine utilization 等指标，验证正确性。
  5. Summarizer agent：从 slow-fast kernel pairs 中提炼通用优化策略（如 "Loop Invariant Code Motion"），生成 experience item 存入 optimization memory。
  6. Beam Search 选择 Top-B kernels 进入下一轮迭代。T=16 轮后输出最优 kernel。
  输出：优化后的 NKI kernel 源码，在 NKIBench 14 个 kernel 上平均 peak throughput 从 49% 提升到 61% (Trainium 1)。
