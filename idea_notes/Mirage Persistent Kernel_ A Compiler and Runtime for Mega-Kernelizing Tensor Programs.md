## Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs

- baseline方法是什么？
  baseline是kernel-per-operator LLM serving系统（SGLang、vLLM、PyTorch + CUDA Graphs + torch.compile），其执行方式为：将模型计算图（DAG of tensor operators）的每个节点作为独立GPU kernel launch。每个operator使用不同的专用kernel库：FlashInfer或FlashAttention处理attention、cuBLAS或cuTLASS处理MatMul、CUDA或Triton处理其余operators。GPU runtime在每个连续kernel launch之间隐式插入kernel barrier，强制前一个kernel的所有thread blocks完成后才能启动下一个kernel。CUDA Graphs用于减少kernel launch overhead但本质上是static capture，对dynamic batch size/shape变化需要重新实例化。

  全栈执行例子（以SGLang/vLLM在H100上执行Qwen3-8B单batch decode iteration为例）：
  - 算法层：标准Transformer decoder——Q/K/V projection (MatMul) → FlashAttention → O projection (MatMul) → RMSNorm → Gate/Up projection (MatMul) → SiLU → Down projection (MatMul) → RMSNorm → ... → LM Head (MatMul)。每个operator对应一个独立kernel。
  - 系统框架层：SGLang/vLLM通过PyTorch执行模型forward。CPU端执行：(1) continuous batching——从request queue取出batch，(2) page allocation——为KV cache分配物理页，(3) 逐operator dispatch kernel launch。CPU-GPU同步发生在每个iteration边界——CPU等待GPU完成上一步后才调度下一步。
  - 编译框架层：无统一编译框架。FlashInfer（Triton/CUDA手写attention kernel）、cuBLAS/cuTLASS（闭源GEMM库）、Triton（JIT编译element-wise/reduction ops）。各库互不感知，无法进行跨算子优化。
  - kernel调度层：每个kernel launch后GPU SM执行SPMD，kernel barrier强制所有SM完成当前kernel → CUDA runtime launch next kernel → 重复。关键瓶颈：(a) kernel launch overhead——每次decode iteration有数百个kernel launch（CUDA Graphs可降低但不能消除），(b) pipeline bubble——kernel barrier阻止跨算子pipelining，TMA/Tensor Cores/CUDA Cores在kernel边界产生pipeline bubble，(c) 粗粒度依赖——AllReduce必须等整个MatMul完成，即使每个AllReduce thread block只依赖一个MatMul thread block的输出，(d) CPU-side scheduling延迟——page allocation和request scheduling在CPU执行，每次iteration需要CPU→GPU dispatch round-trip。
  - 硬件架构层：NVIDIA H100 GPU（132 SMs, HBM 1.6TB/s）。TMA（Tensor Memory Accelerator）、Tensor Cores（989 TFLOPS FP16）、CUDA Cores（60 TFLOPS）三种异构计算单元。在kernel-per-operator模式下：单个MatMul kernel期间Tensor Cores工作但TMA在prefetch完成后闲置、CUDA Cores闲置；attention kernel期间情况类似；AllReduce期间NVLink/PCIe工作但SM闲置。硬件利用碎片化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Mirage Persistent Kernel (MPK)，第一个自动将多GPU模型推理编译为单个mega-kernel的compiler+runtime系统。核心理念是以SM-level graph representation (tGraph)替代kernel-level computation graph，从而暴露传统kernel barrier遮蔽的细粒度并行性。

  解决baseline缺陷的具体设计映射：

  **缺陷1: Kernel barrier阻止跨算子pipelining** → MPK方案：SM-level tGraph + Cross-task software pipelining
  - MPK将每个operator分解为SM级tasks，通过fine-grained event同步替代kernel barrier
  - Cross-task pipelining将每个task拆分为pre-loading phase和compute phase，当前task compute期间即启动下一task的pre-loading
  - Paged shared-memory abstraction让task按需acquire/release shared memory pages，使跨task数据prefetch成为可能
  - 效果：消除pipeline bubble，1.2-1.3x speedup（Qwen3-8B final linear layer on B200）

  **缺陷2: 粗粒度依赖阻止细粒度compute-communication overlap** → MPK方案：Fine-grained dependency analysis
  - MPK在task对级别而非operator级别分析数据依赖——仅当task t1的输出区域与task t2的输入区域重叠时才插入event
  - 例如MatMul→AllReduce：每个AllReduce task仅依赖一个对应MatMul task的输出，MPK通过per-task-pair events使compute和communication在不同SM上并发执行
  - 效果：1.1x speedup（Qwen3-1.7B on 4×H100 TP）

  **缺陷3: Kernel launch overhead + CPU-side scheduling延迟** → MPK方案：In-kernel parallel runtime
  - 整个模型编译为单个mega-kernel，仅一次kernel launch → 消除数百次launch/iteration
  - Page allocation和request scheduling逻辑全部嵌入mega-kernel内部作为单个task执行，消除CPU-GPU同步
  - Workers/schedulers使用circular buffer + atomicAdd实现轻量级task调度

  **缺陷4: 动态workload imbalance** → MPK方案：Hybrid JIT+AOT task launch
  - 对data-dependent执行时间的operator（如attention）使用JIT launch，runtime动态分配实现负载均衡
  - 对确定性的operator（如MatMul after barrier）使用AOT launch，预分配消除dispatch开销
  - MoE场景额外使用hybrid workload balancer：编译期静态分配expert tasks + 运行期根据router产生的global metadata动态调整

  **缺陷5: 分散的kernel库生态** → MPK方案：Compiler-based unified code generation
  - Mirage superoptimizer自动生成每个task的优化CUDA实现，无需手写不同库的kernel
  - NVSHMEM统一处理in-kernel inter-GPU通信
  - 用户只需 torch.compile(backend=MPK)，无需了解底层kernel实现

  全栈执行例子（以MPK在H100上执行Qwen3-8B单batch decode iteration为例，对比baseline）：
  - 算法层：与baseline相同模型架构。MPK不改变模型算法，仅改变执行方式。Attention仍使用paged attention算法，但attention tasks以JIT模式执行以适应sequence-length变异性。
  - 系统框架层：PyTorch + torch.compile(backend=MPK)。编译期：MPK compiler读入计算图 → operator decomposition (H100: ~132 tasks/MatMul) → dependency analysis (per-task-pair events) → event fusion (successor+predecessor) → tGraph normalization (fan-in/fan-out≤1) → tGraph linearization (BFS, 连续task索引) → task code generation (Mirage superoptimizer)。生成的mega-kernel为单个callable PyTorch function。执行期：单次kernel launch，in-kernel runtime持久运行直至所有decode iteration完成。Page allocation、request admission、KV-cache update均在mega-kernel内的start event task中执行——无CPU参与。
  - 编译框架层：MPK编译器完全自研。核心数据结构tGraph存储为GPU device memory compact格式：每个task 352 bytes（dependent_event index, trigger_event index, input/output tensor ptrs, config params）；每个event存储required trigger count + [first_task, last_task] index range。所有tasks和events以连续数组存储，enqueue/dequeue仅用atomicAdd。
  - kernel调度层（核心差异）：SM物理分区——128 workers (每SM一个) + 4 scheduler-SMs (16 warp-schedulers)。以Q_proj → K_proj → Attention → O_proj → RMSNorm → Gate/Up → SiLU → Down → RMSNorm为例：(a) Start event → scheduler dispatch所有Q/K/V projection tasks (AOT, 预分配到workers)，worker SM_i执行Q_proj: TMA preload weight tile → Tensor Core MMA (Q=X×W_Q) → 同时prefetch K_proj weight tile (cross-task pipelining) → 完成, notify event。(b) 所有Q/K/V tasks完成后event激活 → scheduler dispatch attention tasks (JIT, 因为attention执行时间data-dependent)。Worker SM_j执行attention task (JIT): 有long sequence的worker慢、short sequence的快 → 快的worker先完成先获得下游O_proj tasks (JIT) → dynamic load balance。(c) 所有attention tasks完成后barrier event → 后续MLP tasks全部AOT预分配，worker SM_k check AOT queue: event已激活? → execute Gate_proj GEMM + 同时prefetch Up_proj weight → SiLU → Down_proj GEMM + 同时prefetch下一层Q_proj weight → 流水线无缝衔接。(d) 跨全部层，TMA、Tensor Cores、CUDA Cores三种硬件持续饱和——任意时刻都有SM在进行计算、数据搬运或通信。
  - 硬件架构层：NVIDIA H100 GPU。与baseline key hardware utilization差异：Mega-kernel执行全程TMA持续prefetch（消除pipeline bubble），Tensor Cores几乎持续执行GEMM/Attention MMA（各SM间轮转），CUDA Cores持续执行element-wise ops和dequantization（若有），NVLink持续传输（fine-grained overlap with compute）。Kernel-per-operator的理论下限~10ms（纯粹模型参数加载时间），MPK达到12.5ms，只比理论下限高~25%。Baseline SGLang/vLLM为14.5ms，差距主要来自kernel launch overhead和pipeline bubble。

  总结：MPK通过将抽象层次从"kernel级"下沉到"SM级"（即用tGraph替代computation graph），暴露了传统GPU编程模型中被kernel barrier遮蔽的细粒度并行性，并通过compiler自动化和in-kernel runtime高效利用这些并行性，实现end-to-end 1.0-1.7x加速。
