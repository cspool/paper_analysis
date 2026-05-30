## Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs

- 属于编译框架的实现是什么？实验比较什么？
  实现是MPK编译器，将多GPU模型推理的整个计算图自动转换为单个高性能mega-kernel（persistent kernel）。编译器核心流程：(1) Operator decomposition——将每个算子按输出张量维度tiling分解为SM级tasks，各task计算输出张量的不相交子集；(2) Dependency analysis——对共享张量的任意两个算子，枚举task对并引入event作为同步点，event仅在task t1的输出区域与task t2的输入区域重叠时才插入；(3) Event fusion——通过successor-set fusion（合并OutTasks相同的event）和predecessor-set fusion（合并InTasks相同的event）消除冗余同步点；(4) tGraph normalization——通过引入空dummy task将每个task的fan-in和fan-out压缩至最多1个dependent event和1个triggering event，确保task描述的uniform representation；(5) tGraph linearization——使用BFS-based算法（Algorithm 1）线性化tGraph，保证同一event触发的所有task在task队列中连续排列，从而用first/last index紧凑编码fan-out；(6) Task implementation generation——利用Mirage superoptimizer对每个task自动搜索最优thread block graph并生成优化CUDA实现（含intra-SM software pipelining、register reuse、layout优化避免bank conflict）。

  实验比较了MPK与两种SOTA kernel-per-operator LLM serving系统（SGLang、vLLM）和PyTorch baseline（CUDA Graphs + torch.compile）。评估五类模型（Qwen3-8B、Qwen3-1.7B、Qwen3-30B-A3B、Llama-3.1-8B、DeepSeek-R1-Distill-1.5B）在A100/H100/B200上的吞吐量，batch size 1-16。消融实验评估了cross-task pipelining和compute-communication overlap各自贡献。

- 硬件平台是什么，配置是什么。
  三款NVIDIA GPU：A100（108 SMs, 共享内存总容量来自每个SM的配置，32KB page size → 5 pages/SM）、H100（132 SMs, 7 pages/SM）、B200（148 SMs, 7 pages/SM）。多GPU实验：NVIDIA H100 DGX实例（8×H100），使用tensor model parallelism。运行时配置：预留4个SM给scheduler（共16 scheduler warps），其余SM分配给workers。所有GPU使用32KB共享内存page size。精度：bfloat16。

- 开源编译框架是什么。修改了什么。
  开源编译框架是基于PyTorch的kernel backend，通过 torch.compile(backend=MPK) 调用。MPK未修改现有编译框架，而是自研全新编译器（~40K行C++、84K行CUDA、10K行Python）。编译器集成Mirage superoptimizer（用于生成CUDA task实现）和NVSHMEM（用于in-kernel inter-GPU通信）。编译器完全自研，其tGraph representation、event fusion、graph normalization、linearization等均为首次提出。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源链接：https://github.com/mirage-project/mirage

  作用：MPK编译器的核心价值是将PyTorch模型自动mega-kernel化——用户只需 torch.compile(backend=MPK) 即可以少量代码修改将模型编译为单个persistent kernel，消除kernel launch overhead、启用cross-operator software pipelining和细粒度compute-communication overlap，同时保持PyTorch开发体验。

  全过程（以LLaMA-3.1-8B在H100上单GPU推理为例）：
  ```
  输入：PyTorch模型定义 (torch.compile(backend=MPK))
    ├── Attention模块: Q/K/V projection (MatMul) → Attention → Output projection (MatMul)
    ├── MLP模块: Gate/Up projection (MatMul) → SiLU → Down projection (MatMul)
    ├── RMSNorm, Residual Add 等 element-wise ops
    └── Inference配置: batch_size=1, prompt_len=64, decode_len=1024

  MPK编译器流程:
  Step 1 — Operator Decomposition:
    每个MatMul按输出维度tiling为与SM数成比例的tasks数
    (例如 H100: ~132 tasks per MatMul operator)
    输出tile选择策略: 最小化device memory→shared memory数据加载量

  Step 2 — Dependency Analysis:
    枚举所有跨算子task对 (t1→t2)，检查输出/输入区域重叠
    引入event e连接有数据依赖的task对: t1→e→t2

  Step 3 — Event Fusion:
    successor-set fusion: 合并OutTasks相同的events
    predecessor-set fusion: 合并InTasks相同的events

  Step 4 — tGraph Normalization:
    对fan-out>1的task Ti→{e1,...,ek}: 引入dummy task Ti'和中间event e'
      使其变为 Ti→e'→{T1',...,Tk'}→{e1,...,ek}
    对fan-in>1的task 类似处理，最终每个task有≤1 dependent event和≤1 triggering event

  Step 5 — tGraph Linearization:
    执行Algorithm 1 BFS线性化: E←{events with no dependent tasks}
    while E不为空: dequeue event e → 将所有依赖e的task入队T
      → 若某触发event的所有前置task均已入队，该event入队E
    保证同event触发的所有task在T中连续排列 → 用[first, last]索引编码fan-out

  Step 6 — Task Code Generation:
    每个task关联一个PyTorch参考实现
    Mirage superoptimizer搜索最优thread block graph
    Mirage编译器生成优化CUDA device function:
      - intra-SM software pipelining (TMA + Tensor Cores + CUDA Cores)
      - register reuse, shared memory layout optimization

  输出：单个mega-kernel (callable PyTorch function)
    包含:
    - 线性化tGraph (存储在GPU device memory)
      · 每个task: dependent_event索引, trigger_event索引
      · 每个event: 所需trigger次数, [first_task, last_task]索引范围
    - 所有task的优化CUDA device functions
    - 每个task描述符 (352 bytes): input/output tensor指针, 配置参数
    - In-kernel runtime embedded code: workers, schedulers, task queues

  执行: 调用该function → 单次GPU kernel launch
    - 各SM worker: dequeue task → execute → notify trigger_event
    - 各scheduler warp: poll event激活 → dispatch tasks to workers
    - 直至所有tasks完成，mega-kernel返回
  ```
