## Leftover Policy (GPU Thread Block Scheduling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Leftover Policy 是 NVIDIA GPU thread block scheduler 决定**when/which** thread block 被调度的策略。该 policy 规定：只有**当前执行队列头部 kernel**的 thread block 可以被调度到 SM 执行；在队头 kernel 的所有 block 都被调度完之前，队列中其他 kernel 的 block 不会被考虑。这个 policy 被称为 "leftover" 是因为它总是先清空（finish off）当前 kernel 的剩余 block，再处理下一个 kernel。Leftover policy 由先前工作（Naghibijouybari et al. [11]、Amert et al. [2]、Li et al. [10] 等）首次观察到，本论文验证 Pascal/Volta/Turing 均采用此策略。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Leftover Policy 与 Most-Room Policy 的协作逻辑：

```
GPU Execution Queue:
  [Kernel A: 100 blocks] [Kernel B: 50 blocks] [Kernel C: 30 blocks]
    ▲队头

Thread Block Scheduler 执行流程:
  Step 1: Kernel A 是队头
    - 调度 A.block_0, A.block_1, ..., A.block_99
    - B 和 C 的 block 均不被调度，即使 SM 有空闲资源

  Step 2: Kernel A 所有 block 调度完毕
    - 移除 A 出队列
    - Kernel B 成为新队头
    - 开始调度 B.block_0, B.block_1, ...

  Step 3: B 的 block 可能被 colocated 到已有 A block 的 SM
    - Most-Room Policy 决定具体放置
    - 如果 B 的 block 数量小（所有 block 可一次全部调度），
      则 A 和 B 的 block 会在 SM 上共存 → 真正的并发执行
```

Leftover policy 的关键影响：
- **小 kernel 并发**：如果 workload 由多个小 kernel 组成（每个 kernel 的 block 数 ≤ SM 总容量），leftover 允许队头 kernel 的 block 全部被调度后，下一个 kernel 的 block 立即开始与前一 kernel 的末尾 block 并发执行
- **大 kernel 独占**：如果队头 kernel 是大 kernel（block 数超过 SM 总容量），其 block 分批(wave)调度，但队列中后续 kernel 始终没有机会——直到该大 kernel 全部完成，GPU 被独占
- **不可抢占**：Leftover + 不可抢占意味着一旦大 kernel 获得 SM 资源，小 kernel 无法插入

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

这是 NVIDIA GPU 硬件调度器的固有行为，非用户可配置。应用开发者通过以下方式间接管理：(i) 控制 kernel launch 顺序（先发射小 kernel vs 先发射大 kernel）；(ii) 使用 CUDA MPS 创建独立的 GPU 分区；(iii) 使用 CUDA stream priority hints（`cudaStreamCreateWithPriority`）。论文建议 GPU 模拟器在调度模型中加入 leftover policy 以提高模拟精度。

涉及论文标题：
- Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
