## KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

- 属于编译框架的实现是什么？实验比较什么？
  实现是KernelEvolve——一个基于LLM Agent的自动化kernel生成与优化框架。核心包括：(1) 图搜索状态机（Graph-Based Search），支持greedy search、Monte Carlo Tree Search (MCTS)和evolutionary algorithms三种搜索策略；(2) 通用算子（Universal Operator），通过retrieval-augmented dynamic prompting动态合成prompt，替代传统的多算子（Draft/Debug/Improve）静态模板方案；(3) Deep Search Sub-Agent——从持久化知识库（hierarchical file system）中检索硬件约束（constraints/）、优化指南（guidance/）和平台特定文档（hardware/{nvidia|amd|mtia}/）；(4) Context Memory Sub-Agent——分析运行时profiling结果并合成动态prompt，维护metadata store（关系数据库）+ object store（kernel实现文件）的两层存储架构；(5) MTIA Knowledge Injection——将MTIA专有架构知识（SFU、inter-PE communication、dual-core synchronization、custom type system）注入知识库以教育LLM；(6) Evaluation and Tooling Framework——包含TritonBench correctness验证、Torch Profiler系统级profiling、NCU/Proton kernel级profiling、Triton MPP统一profiling、MTIA Insight硬件特定profiling，以及FaaS-based remote evaluation。

  实验比较：
  - OSS Operator Evaluation（Section 4）：160个ATen operators × 3平台（NVIDIA H100, AMD MI350, MTIA v3）= 480个operator-platform配置，验证100% correctness。KernelBench Level 1/2/3共250个问题100% pass rate。6个代表性operator的fitness score trajectory（50 steps）。
  - Convolutional Transformer（Section 5.1）：conv1d kernel vs PyTorch conv1d和conv2d baseline，在NVIDIA H100上FP16/FP32精度，多种batch size和shape。
  - Cross-Platform Convolution（Section 5.2）：conv1d kernel跨5个硬件平台（NVIDIA H100, A100, AMD MI300, MI350, MTIA v3）的性能对比。
  - WuKong Optimized FM（Section 5.3.1）：fused batched matrix multiplication kernel vs PyTorch torch.compile baseline，在NVIDIA H100上多种production shapes。
  - InterFormer PFFN（Section 5.3.2）：fused FFN+GELU+RMSNorm kernel vs PyTorch torch.compile baseline，在NVIDIA H100上多种production shapes。
  - MapIdTransform（Section 5.4.1）：fused data preprocessing kernel vs PyTorch baseline，在MTIA v2i和MTIA v3上多种batch size和mapping table size。
  - MergeBucketizedDense Transform（Section 5.4.2）：fused bucketization kernel vs PyTorch baseline，在MTIA v2i和MTIA v3上多种Batch × Features × Borders配置。
  - Batch Event Truncate（Section 5.5）：batched Triton kernel vs non-batched PyTorch baseline，在MTIA上多种feature counts和event lengths配置。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU（Hopper架构，Tensor Memory Accelerator, WGMMA, mbarriers），NVIDIA A100 GPU（Ampere架构），AMD MI300 GPU（Infinity Cache），AMD MI350 GPU，MTIA v2i（8×8 PE array, dual RISC-V cores per PE, SFU/DPE/RE/SIMD/MLU/CP fixed-function units），MTIA v3（next-gen MTIA）。LLM backends：Claude 4.5、GPT-5、Meta's CWM、Llama on Twine。软件环境：PyTorch + Triton + Triton-MTIA + Triton TLX，TritonBench validation framework，Meta's Bento Jupyter notebook platform，Conveyor continuous deployment system，FaaS（Function-as-a-Service）platform for remote kernel evaluation。Profiling工具：TritonBench、Torch Profiler、NCU (NVIDIA)、Triton Proton、Triton MPP (Multi-Pass Profiler)、MTIA Insight。

- 开源编译框架是什么。修改了什么。
  论文未提供KernelEvolve框架本身的完整开源链接。系统基于多个开源组件构建：
  - Triton (https://github.com/triton-lang/triton)：作为主要的kernel编程DSL，KernelEvolve扩展了Triton的MTIA backend（Triton-MTIA），包括libdevice API映射（SFU LUT operations）、cross-PE communication primitives（tl.load/tl.store with direction attribute）、runtime barriers（tl.pe_runtime_barrier）、custom type system（TensorView, CoreID, ExecutionGrid）、MTIA-specific compilation options（cb_multiplier, use_dual_core）。
  - Triton TLX (https://github.com/facebookexperimental/triton)：用于NVIDIA Hopper的低级优化（warp specialization, async tensor core operations）。
  - TritonBench (https://github.com/meta-pytorch/tritonbench)：用于kernel correctness验证和speedup测量。
  - KernelBench：用于benchmark LLM kernel生成能力。
  - KernelAgent (https://github.com/meta-pytorch/KernelAgent)：多agent kernel合成系统。
  - KernelLLM (https://huggingface.co/facebook/KernelLLM)：Triton kernel生成模型。
  - MLIR-based multi-target compilation pipeline：Triton-MLIR → GPU/AMDGPU/MTIA dialects → LLVM-IR → PTX/CUBIN (NVIDIA) / AMDGCN/HSACO (AMD) / RISC-V (MTIA)。

  KernelEvolve的核心修改是：在Triton编译框架之上构建了完整的agentic kernel生成pipeline——将kernel优化建模为图搜索问题，通过retrieval-augmented动态prompt synthesis替代静态编译策略，并在持久化知识库中编码硬件特定约束。具体包括：(1) 通用算子替代Triton编译器的固定优化pass；(2) 知识库驱动的硬件awareness替代编译器内置的cost model；(3) 多层次profiling feedback（系统级→kernel级→intra-kernel级）作为搜索fitness信号；(4) FaaS-based remote evaluation将kernel评估从本地编译器中解耦。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  KernelEvolve框架本身是Meta内部生产系统，论文未提供完整开源链接。但系统基于多个开源组件构建（见上）。以下是基于论文描述的全过程：

  作用：KernelEvolve将kernel开发从手动、专家依赖的过程转变为自动化、可扩展的AI agent服务——输入kernel specification（算子类型、输入shape、目标硬件平台），输出经过正确性验证和性能优化的Triton kernel实现。核心理念是"用LLM agent的inference-time scaling替代人工tuning"。

  全过程（以NVIDIA H100上生成conv1d kernel为例）：
  ```
  输入：用户提供kernel specification
    - 算子类型: conv1d (1D convolution)
    - 生产shape: (B=2048, Cin=96, Cout=96, L=200)
    - 目标平台: NVIDIA H100 (Hopper)
    - 精度: FP16
    - Baseline: PyTorch torch.nn.functional.conv1d + conv2d workaround

  Step 1 - 初始化搜索图 (Section 3.1):
    → 创建root node v0 = PyTorch baseline specification
    → 初始化搜索图 G0 = ({v0}, ∅)
    → 选择搜索策略: 论文未明确说明conv1d所用策略（300 steps搜索，fitness = 1/latency）

  Step 2 - 迭代搜索循环 (Step t):
    → Selection: 通过selection policy π_sel 选择待扩展节点（greedy/MCTS/evolutionary）
    → Prompt Synthesis:
        Context Memory Sub-Agent:
          - 读取当前kernel源码和execution history
          - 分析profiling结果（TritonBench speedup, Torch Profiler timeline, NCU metrics, MPP instruction-level traces）
          - 生成bottleneck诊断报告（如：5次kernel launch + layout transform overhead）
          - 合成优化指令（如：fuse all operations into single kernel, eliminate NCHW↔NHWC conversions）
        Deep Search Sub-Agent:
          - 根据诊断报告检索知识库：
            → hardware/nvidia/arch/tensor_cores.md (Tensor Core capabilities)
            → hardware/nvidia/optimization/tma.md (TMA for async transfers)
            → hardware/nvidia/tlx/warp_specialization.md (warp specialization)
            → guidance/performance/autotuning.md (block size, warp count, pipeline stages)
            → guidance/triton/memory_primitives.md (cache modifiers .ca/.cg)
            → code_samples/hopper-gemm-pipelined.py (reference implementation)
        Dynamic Prompt合成:
          → 组合: kernel源码 + profiling分析 + 检索到的知识库内容 + hardware约束
          → Token budget管理: 64K-1M tokens (取决于LLM backend)

    → LLM Generation (Universal Operator):
        调用Claude 4.5/GPT-5/CWM生成新的Triton kernel候选
        Kernel输出格式：PytorchModel(nn.Module) + TritonModel(nn.Module) + get_inputs()

    → Evaluation (Section 3.4):
        Evaluation Code Generator → 生成TritonBench harness + Torch Profiler + NCU/MPP scripts
        FaaS dispatch → remote H100 worker执行:
          1. TritonBench: torch.allclose(atol=1e-4, rtol=5e-4) correctness验证
          2. Speedup measurement: t_pytorch / t_triton
          3. Torch Profiler: execution trace (kernel counts, launch overhead)
          4. NCU: occupancy, memory throughput, instruction mix
          5. Triton MPP: instruction-level pipeline behavior
        → 返回结构化结果: {correctness: bool, speedup: float, profiling_metrics: {...}}

    → Graph Update:
        创建新节点v_t+1，记录fitness score F(v) = t_pytorch / t_triton
        错误/correctness失败 kernel: F(v) = 0
        持久化: metadata store (id, pid, score, is_buggy, path_ref) + object store (kernel_n.py, overview.md)

  Step 3 - 终止条件 (Termination Rule τ):
    → 达到wall-clock time budget 或 max artifacts数量
    → progress stall（连续N步无fitness提升）
    → fitness threshold达到（如speedup > 2.0×）

  Step 4 - 发现的关键优化（Section 5.1分析）:
    经过300步搜索，KernelEvolve自动发现:
    1. Kernel Fusion: 将5个独立kernel（nchwToNhwcKernel × 2 + GEMM + nhwcToNchwKernel + fused_convolution）融合为2个kernel（pack_conv1d_weight_kernel + conv1d_gemm_kernel）
    2. Expended Autotuning: 20+ autotune configurations (BLOCK_M ∈ {32,64,128}, BLOCK_N ∈ {32,64,128}, BLOCK_K=32, num_warps ∈ {2,4,8}, num_stages ∈ {2,3,4})
    3. 3D Grid Launch: 并行化grouped convolution channels, 消除inter-group dependencies
    4. Double-Buffered Execution: prefetch next data blocks while computing current
    5. Differentiated Cache Modifiers: .ca for streaming activations, .cg for reused weights

  输出：经过正确性验证、性能优化、具备生产就绪的Triton conv1d kernel
    - Fitness score: 6889 (从初始~2000经300步搜索提升)
    - Speedup: 2.30× vs PyTorch conv1d, 1.62× vs PyTorch conv2d (FP16, production shape)
    - 跨平台speedup: 1.75× (MI300), 2.54× (MI350), 1.77× (A100), 6.54× (MTIA v3)
  ```
