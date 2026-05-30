## Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels

- baseline方法是什么？
  **Round-robin scheduling assumption（先前的共识假设）**：在并发 kernel 执行场景下，GPU thread block scheduler 使用 round-robin policy 将 thread block 分配到 SM。此前的研究（GPGPU-Sim、Accel-Sim 等 GPU 模拟器，以及 Naghibijouybari et al. [11]、Amert et al. [2]、Li et al. [10] 等工作）均假设或观察到 thread block scheduler 使用 round-robin 策略轮询地将 block 分配到各个 SM。

  全栈执行例子（以 Turing GPU 上两个并发 kernel 的 round-robin 假设执行流程为例）：
  - **模型推理算法层**：论文未涉及 ML 推理。使用 purpose-built kernel 类别（L1-cache-dependent、compute-intensive、memory-intensive、transfer-bandwidth-dependent）。
  - **系统框架层**：标准 CUDA programming model。kernel 从不同 CUDA stream 发射，thread block scheduler 负责将 block 分配到 SM。Round-robin 假设下，scheduler 依次将 block 分配到 SM0, SM1, SM2, ...
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Round-robin policy 假设：当 Kernel X 的 5 个 block 先占满 5 个 SM 后，Kernel Y 的第一个 block 释放时，scheduler 将 Kernel Y 的 block 依次轮询分配到 SM0 → SM1 → SM2 → SM3 → SM4 → SM0 → ...，按固定循环顺序，**不考虑各 SM 当前的资源可用性差异**。在单 kernel 场景中，由于所有 block 大小相同且行为相似，round-robin 与 most-room 的表现难以区分（各 SM 资源可用性基本相同）。
  - **硬件架构层**：标准 NVIDIA GPU（Pascal/Volta/Turing），thread block scheduler 是 NVIDIA 闭源硬件实现的 black-box。

  Baseline 缺陷（round-robin assumption 的问题）：
  - (a) **Round-robin 假设在并发 kernel 场景下是错误的**：当存在多个不同尺寸的 kernel 并发执行时，各 SM 的资源可用性因已resident 的 block 大小不同而异，round-robin 无法解释实际的 placement 行为。
  - (b) **GPU 模拟器精度受损**：GPGPU-Sim、Accel-Sim 等模拟器若假设 round-robin，在模拟并发 kernel workload 时会得出错误的 block distribution，导致性能预测不准。
  - (c) **无法预测"反直觉"的性能退化**：例如减少 1 个 thread/block 导致 3.58X execution time 增加（transfer-bandwidth-dependent kernel），这种看似矛盾的现象无法用 round-robin 解释。
  - (d) **缺乏对 concurrent kernel 性能的系统性理解**：现有调度研究关注 time-multiplexing（preemption）或 space-multiplexing（resource sharing），但均基于对硬件 scheduler 的不完全认知。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **通过实证测量推导 Most-Room Policy 并表征其性能影响**：论文通过设计精确控制 block 执行时间和 resource dimension 的两 kernel 实验，从真实硬件的 behavior 推导出 thread block scheduler 的 "most-room policy"。核心发现：scheduler 选择能容纳当前 kernel 最多 block 数量的 SM 来放置下一个 block，按 pre-defined device-specific ordering 打破平票。论文进一步设计了四类 purpose-built kernel 来表征 most-room policy 在不同 kernel 类别下的性能影响。

  全栈执行例子（以 most-room policy 推导实验，Pascal GPU，Figure 2 为例）：
  - **模型推理算法层**：论文未涉及 ML 推理。
  - **系统框架层**：标准 CUDA stream concurrency。Kernel X 从 stream 1 发射（5 blocks × 256 threads），Kernel Y 从 stream 2 延迟发射（3 blocks × 160 threads）。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：
    1. **时序控制**：通过 `globaltimer` spin 使 Kernel X 的 block 按 SM id 顺序完成（B0 on SM0 先完成 → B4 on SM4 最后完成），在 SM0 空但 SM1-4 仍 busy 时发射 Kernel Y。
    2. **Most-room 决策**：Scheduler 计算每个 SM 可容纳的 Kernel Y block 数 = floor(SM 剩余 threads / Y.threads_per_block)。SM0 空（2048 free = 12 blocks），SM1-4 各含一个 X block（1792 free = 11 blocks）→ scheduler 选 SM0（most room）→ Y0→SM0。
    3. **资源重算**：Y0 占 160 threads 后，SM0 剩余 1888 = 11 blocks，与 SM1-SM4 平票 → tie-breaking order → Y1→SM0。
    4. **再次重算**：Y1 占后 SM0 剩余 1728 = 10 blocks，SM1-4 各有 1792 = 11 blocks → Y2→SM1。
    5. **结果**：Y0,Y1→SM0, Y2→SM1（most-room），而非 round-robin 的 Y0→SM0, Y1→SM1, Y2→SM2。
  - **硬件架构层**：通过 `smid` 寄存器读取 SM id，`blockIdx` 识别 block。实验揭示 limiting resources 包括 threads、shared memory、blocks/SM、warps/SM（论文声明可能还有其他未识别的因素）。

  性能影响表征（以 Turing GPU, L1-cache-dependent kernel 为例）：
  - **kernel调度层**：通过改变 Kernel B 的 threads/block（33 vs 32），触发 different limiting resource：33 threads → limiting=threads → all B blocks 分配到唯一的空 SM67（concurrent-isolated）；32 threads → limiting=blocks/SM → B blocks 分布到 8 个 SM，部分与 A 的 blocks colocate（concurrent-colocated）。
  - **性能结果**：1 thread/block 的不同导致 colocation，L1 cache contention 使 Kernel A 从 85ms→105ms (1.24X)、Kernel B 从 79ms→105ms (1.33X)，total time 从 85ms→105ms。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: round-robin 假设错误** → 方案：通过 `globaltimer` 时序控制 + `smid` 位置追踪的实证实验设计，精确揭示 most-room policy。当 SM 间的资源可用性因已 resident block 尺寸不同产生差异时，most-room 与 round-robin 的行为明显不同（Figure 2），而单 kernel 场景两者几乎无区别（Section 4.4 解释 why round-robin 被长期误用）。
  - **defect: GPU 模拟器精度受损** → 方案：提供三种微架构（Pascal/Volta/Turing）的 most-room policy 详细参数，包括 limiting resource 识别（threads、shared memory、blocks/SM、warps/SM）和 device-specific tie-breaking ordering（Pascal=ascending, Turing=even-then-odds, Volta=device-specific），可直接用于改进模拟器中的 thread block scheduler 实现。
  - **defect: 无法预测反直觉性能退化** → 方案：通过四类 purpose-built kernel 的系统性实验（Section 5），展示 most-room policy 如何将微小的 kernel 参数变化（如 1 thread/block）放大为 significant performance degradation（1.33X-3.58X for L1-cache/transfer kernels），并解释 root cause（colocation → resource contention → specific resource type matters）。
  - **defect: 缺乏 concurrent kernel 性能的系统化理解** → 方案：识别出影响 concurrent kernel 性能的三个关键因素：(i) thread block scheduler 的 scheduling policy；(ii) 多种硬件资源（L1 cache、functional units、global memory bandwidth、PCIe bandwidth、TLB）的潜在竞争；(iii) kernel launch timing 等可能不可预测的因素。Section 5 按 kernel 类别分别分析这些因素的影响。
  - **额外发现: leftover policy 与 most-room 配合**：Leftover policy（Section 6）定义 when/which block 被调度（只有队列头 kernel 的 block 可被调度），most-room policy 定义 where 放置该 block。两者共同决定了 concurrent kernel 的调度行为空间。
