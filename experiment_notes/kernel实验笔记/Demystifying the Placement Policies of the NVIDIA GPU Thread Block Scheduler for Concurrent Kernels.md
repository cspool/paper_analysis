## Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  本论文是一个测量/表征研究（measurement/characterization study），而非提出新系统。实现分为两部分：(i) 通过设计特定的两 kernel 实验 workload（Kernel X + Kernel Y），利用 `smid` 寄存器和 `globaltimer` 计时代码，实证推导 NVIDIA GPU thread block scheduler 在并发 kernel 下的调度策略——"most-room policy"；(ii) 设计四类 purpose-built kernel（L1-cache-dependent、compute-intensive、memory-intensive、PCIe-transfer-bandwidth-dependent），测量 most-room policy 对并发 kernel 性能的影响。
  实验比较了三种场景：(i) Serial（串行，baseline——每个 kernel 独立运行，无并发）；(ii) Concurrent-Isolated（并发但 Kernel B 的所有 block 被分配到与 Kernel A 不同的 SM 上）；(iii) Concurrent-Colocated（并发且 Kernel A 和 Kernel B 的 block 被分配到同一个 SM 上）。指标为各 kernel 的 execution time（ms），通过 nvprof 测量。

- 后端平台是什么，配置是什么。
  - **Pascal**: GeForce GTX 1080, Compute Capability 6.0, 5 SMs, 2048 threads/SM, 1024 max threads/block, 32 max blocks/SM, 64 max warps/SM
  - **Volta**: Tesla V100, Compute Capability 7.0, 80 SMs, 2048 threads/SM, 1024 max threads/block, 32 max blocks/SM, 64 max warps/SM
  - **Turing**: GeForce RTX 2080 Ti, Compute Capability 7.5, 68 SMs, 1024 threads/SM, 1024 max threads/block, 16 max blocks/SM, 32 max warps/SM

- 评估性能的软件/脚本是什么。修改了什么。
  论文使用自编 CUDA kernel 而非标准 benchmark（如 Rodinia），以精确控制调度结果和特定资源的竞争。使用 NVIDIA nvprof 作为性能 profiling 工具测量 kernel execution time。
  - **L1-cache-dependent kernel**：每个线程反复访问 texture memory，利用各 GPU 的 L1 cache 大小和 set-associativity 信息使访问高 cacheable 但易被替换。测量到 serial case 下 L1 cache hit rate 平均 90%（75%-95%）。
  - **compute-intensive kernel**：反复执行浮点运算占用 functional units，避免 global memory 访问以防止内存竞争影响结果。
  - **memory-intensive kernel**：反复写入 global memory 中的大数组（使用写操作防止 L1/texture cache 缓存），线程间访问地址间隔拉开以避免 coalescing。
  - **transfer-bandwidth-dependent kernel**：利用 UVM（Unified Virtual Memory）触发大量 page fault，通过 PCIe 异步传输数据。线程内 block 的地址靠近以允许 coalescing，block 间地址远离以减少 global memory contention 对 PCIe 竞争的影响。
  - 调度策略推导 kernel：使用 `globaltimer` 寄存器让每个 block spin 与 SM id 成比例的时间（B0 最短，Bn 最长），精确控制 block 完成顺序，暴露 scheduler 的 placement 决策。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供开源链接。这是一篇 2020 年左右的工作（发表于 SIGMETRICS 相关 venue），作者来自 Worcester Polytechnic Institute。论文使用 Google Cloud Platform 运行实验，使用 NVIDIA profiler nvprof。

  评估原理（以 Turing GPU L1-cache-dependent kernel 的 most-room policy 性能影响实验为例）：
  1. **输入设计**：Kernel A 发射 n−1=67 个 block（Turing 有 68 SM），每个 block 512 threads，先发射到第一个 CUDA stream。保证所有 67 block 各自占据一个 SM（SM0-SM66），SM67 空置。
  2. **Kernel B 变体设计**：Kernel B 有两个版本——version1 每 block 33 threads（concurrent-isolated 条件），version2 每 block 32 threads（concurrent-colocated 条件）。两者均 8 blocks，从第二个 CUDA stream 晚于 Kernel A 发射。
  3. **Most-room policy 触发**：33-thread 版本以 threads 为 limiting resource → SM67（空 SM）可容纳最多 Kernel B block → 全部 8 block 均分配到 SM67（concurrent-isolated）。32-thread 版本以 blocks/SM 为 limiting resource → SM67 分配第 1 个 block 后，所有 SM 可容纳的 block 数相同 → 按 tie-breaking 顺序（even-then-odds: 0,2,4,...,66,1,3,...,67）分配剩余 block → 部分 Kernel B block 与 Kernel A block colocate（concurrent-colocated）。
  4. **Kernel 执行**：Concurrent-isolated 时 Kernel A 和 Kernel B 在完全独立的 SM 上执行，各自独占 L1 cache。Concurrent-colocated 时部分 SM 上两个 kernel 的 block 共享 L1 cache，产生 cache contention。
  5. **性能测量**：nvprof 测量各 kernel execution time。每秒测量 30 次取平均，coefficient of variation < 3%。比较 serial（各 kernel 独立运行无并发）vs concurrent-isolated vs concurrent-colocated。
  6. **输出**：Concurrent-isolated 下 Kernel A=85ms, Kernel B=79ms（与 serial 一致）。Concurrent-colocated 下 Kernel A=105ms (1.24X), Kernel B=105ms (1.33X)。Total time: serial=164ms, isolated=85ms (0.52X), colocated=105ms (0.64X)。

  Most-room policy 推导实验流程（Pascal GPU 示例，Figure 2）：
  1. Kernel X: 5 blocks（Pascal 有 5 SM），256 threads/block。通过 `globaltimer` spin 保证 B0 最先完成（SM0），B4 最后完成（SM4）。
  2. Kernel Y: 3 blocks，160 threads/block。发射时机使得 B0 已完成（SM0 空），B1-B4 仍在运行（SM1-SM4 各有一个 Kernel X block）。
  3. Scheduler 决策：SM0（空, 2048 free threads = 12 blocks of Y）> SM1-SM4（1792 free threads = 11 blocks of Y）→ Y0→SM0。Y0 占 160 threads 后 SM0 剩 1888 threads = 11 blocks → 与 SM1-SM4 平票 → tie-breaking SM0 → Y1→SM0。Y1 占后 SM0 剩 1728 threads = 10 blocks < SM1-SM4 的 11 blocks → Y2→SM1。
  4. 如果 scheduler 是 round-robin，Y0→SM0, Y1→SM1, Y2→SM2。实际观察到 Y0,Y1→SM0, Y2→SM1，证明 most-room 而非 round-robin。
  5. Tie-breaking ordering: Pascal = ascending (0,1,2,3,4); Turing = even-then-odds (0,2,4,...,66,1,3,...,67); Volta 也有独立的 fixed ordering。

  论文还通过修改 limiting resource 的变体实验验证了 other limiting factors：将 Kernel Y threads 降至 32（limiting factor 变为 blocks/SM），观察 block distribution；将 Kernel Y threads 升至 33（limiting factor 变为 warps/SM），观察不同的 placement 行为——均与 most-room policy 预测一致。
