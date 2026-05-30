## Dynamic Expert Placement on GPUs（GPU上的动态专家放置）

术语是什么？
Dynamic Expert Placement 是 ES-MoE 提出的 per-batch expert→GPU 映射调度策略。传统 expert parallelism 中 expert 静态固定在 GPU 上，导致 GPU 间 load imbalance 和大量 zero-padding。ES-MoE 通过 CPU offloading 使 expert 不再固定于 GPU，每个 batch 根据 gating network 输出的 token 分布动态重新分配 expert 到 GPU，最小化 makespan。

从kernel调度角度拆解术语：
贪心调度算法（Graham 1969, 4/3-approximation）：(1) 建模每个 expert 的处理时间 = max(upload_time + compute_time × token_count)；(2) 按处理时间降序排列 experts；(3) 依次将每个 expert 分配给累积负载最小的 GPU。复杂度 O(m log n + m log m)，CPU 执行 < 2.69μs。实验中 GPU 间 token 差异从 102%（Fairseq static）降至 15%（ES-MoE dynamic），完全消除 zero-padding。

术语一般如何实现？如何使用？
实现于 ES-MoE/Fairseq：gating network 执行后 → all-reduce per-expert token counts → CPU 执行 greedy placement → 输出 expert→GPU 映射 → 各 GPU 按映射上传对应 experts。前提是 experts 已 offload 到 CPU。

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization

SmartMoE 对该术语的扩展：SmartMoE 将 Expert Placement 从单纯的运行时调度策略升格为并行策略空间中的一个独立可搜索维度。提出三种 placement 搜索算法：(1) **Greedy**：O(NE)——按 per-expert token count 降序排列，依次将 expert 分配到累积负载最小的 GPU（限制 per-GPU expert 数 ≤ E/N）；(2) **DP**：O(N×4^E)——状态 F(i,S) 表示前 i 个 GPU 已放置 expert 集合 S 的最小 makespan，保证最优解；(3) **Hybrid**：O(ME + N×4^M)——先 Greedy 将 E 个 expert 分配到 M 个虚拟设备，再 DP 将 M 个虚拟设备分配至 N 个物理设备（M 可调，如 M = GPUs_per_node）。SmartMoE 将 Expert Placement 作为整个并行化框架中唯一在线可变的维度，而 DP/TP/PP 在离线阶段固定。
