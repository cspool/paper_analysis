## Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出Difflow（原名ChituDiffusion）运行时kernel层优化，实现ragged batching kernel和冗余消除kernel。核心kernel/运行时实现：(1) Ragged Data-Independent Operation Kernels：实现四个ragged data-independent operation kernel (Triton + CUDA)，用于支持ragged batch请求的高效执行。对ragged data-independent操作（无跨请求共享数据，如transpose、reduce），采用每请求独立并行执行策略——基于已有regular operator的tiling plan和computing microkernels，将每个请求划分为tile集合，通过round-robin policy在batched execution时映射到GPU thread blocks；(2) Ragged Data-Sharing Operation Regularization Kernel：对ragged data-sharing操作（有共享权重，如convolution、linear），通过transpose+reshape/transpose+im2col等图变换kernel将ragged输入compact为regular shape→调用标准kernel库执行（如ragged Matmul通过fuse batch dim和ragged dim→reshape→regular Matmul kernel）；(3) Redundancy Memory Access Elimination Kernel：对attention操作中冗余K/V tensors，运行时压缩K/V沿redundant batch dimension、concat Q tensors from different requests into single one (Figure 6)→使用FlashAttention等标准attention kernel执行压缩后的计算；(4) Invariant Tensor Elimination Runtime：lightweight四态(constant/loop-invariant/loop-variant/unknown)检测→compile-time precompute constants→loop-invariants hoisted→multi-value constants selective fixing (trade off performance vs generation diversity)。实验比较：Ablation study在edit应用上隔离各优化的逐项throughput贡献 (ChituDiffusion-base→+SCH 1.29×→+COMP 1.56×→+IRE 1.71×)；图13(b) sequential execution (无batching)下IRE贡献1.3× speedup；raggedness ratio sweep (0%-100%)验证uniform/ragged/mixed dEngine选择效果。

- 后端平台是什么，配置是什么。
  NVIDIA A100 40GB PCIe GPU (CUDA 12.1) + NVIDIA H100 80GB PCIe GPU (CUDA 12.1 UNet / CUDA 12.8 DiT)。开源release PyTorch 2.9。

- 评估性能的软件/脚本是什么。修改了什么。
  基于Triton[59]和CUDA实现四个ragged data-independent operation kernel。性能模型基于OLS regression (R²=0.998)，profiling 16 samples (batch 1-16, shape 256-768)，96-sample evaluation set (R²=0.996, RMSE <3μs)。修改：在Triton/CUDA层新增ragged operation kernels——round-robin tile-to-thread-block mapping用于data-independent ops、transpose+reshape/im2col用于data-sharing ops的regularization。冗余消除kernel通过等价线性代数变换（compress K/V + concat Q）避免新建kernel直接复用FlashAttention。Invariant tensor detection算法从tensor definitions初始化→iterative propagation with priority hierarchy。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/thu-pacman/chitu/tree/Diffusion。kernel使用流程：
  1. Ragged batching kernel：当scheduler决定使用ragged dEngine执行一批不同shape的requests时→ragged data-sharing ops (Matmul/Conv) 通过transpose/reshape/im2col transform转为regular ops→regular Matmul/Conv kernel执行→ragged data-independent ops (transpose/reduce) 通过round-robin tile mapping在GPU thread blocks并行执行
  2. 以ragged Matmul为例：input [b, m̂, k] (m̂ ragged) + weight [k, n]→transpose+reshape fuse b和m̂→[b·m̂, k] regular Matmul→regular kernel
  3. 冗余内存消除：attention中K/V tensors有相同prompt→沿batch dim compress去重→concat所有请求的Q→标准FlashAttention计算→broadcast恢复
  4. Invariant tensor elimination：编译时detection→constant tensor precomputed/loop-invariant hoisted→multi-value constants在运行时selective fixing

Difflow kernel/运行时的作用：通过ragged operation regularization将异构shape请求转化为kernel-compatible形式（无需手写所有ragged kernel），通过冗余消除规则在tensor代数层面等价去除冗余计算和内存访问，使运行时能够高效利用现有优化kernel库同时支持数据属性感知的优化。

