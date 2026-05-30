## MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing

- baseline方法是什么？
  Baseline 是 Megatron-LM 的默认 expert parallelism，采用 contiguous block 专家放置策略：将每层的 E 个 expert 按索引顺序均匀分配给 G 个 GPU（如 8 experts / 4 GPUs → GPU0 分配 experts 0-1、GPU1 分配 experts 2-3...）。此策略仅考虑内存均衡（每个 GPU 等量 expert 参数），不考虑：**(1) Token 处理负载不均衡**——某些 expert 被激活频率远高于其他 expert（如 layer 14 中 experts 0-1 处理 64% token，layer 23 中 experts 6-7 处理 69% token），导致 hosting GPU 处理时间远超其他 GPU，产生计算尾延迟；**(2) 跨 GPU 通信倾斜**——all-to-all token dispatching 中不同 GPU pair 间通信量严重不均，某些 pair 通信量远超其他，带宽利用不均导致通信尾延迟。在 Mixtral-8x7B 中 all-to-all 通信占端到端推理时间的 35.7%。
  全栈执行例子（Baseline - Megatron-LM contiguous placement）：
  - **算法层**：Mixtral-8x7B，32 层 MoE decoder，每层 8 experts + top-2 routing → 每个 token 在每层由 router 选择 top-2 expert → 输出为两个 expert 的加权和。
  - **Serving/框架层**：Megatron-LM 初始化 → expert parallel size=4 → 每层 experts {0,1}→GPU0, {2,3}→GPU1, {4,5}→GPU2, {6,7}→GPU3（contiguous block）→ 每个 micro-batch 到达 → token dispatch: 每个 GPU 上的 token 按 router 决策 → all-to-all send 到目标 expert 所在 GPU → expert FFN 计算 → all-to-all send 结果回原 GPU。
  - **通信执行**：All-to-all 通过 NCCL group 实现 → intra-node 走 NVLink（900GB/s）、inter-node 走 IB（400/800Gbps）→ layer 14 中 expert 0-1 处理 64% token → GPU0 接收大量 remote token、GPU1-3 接收少 → GPU1-3 完成通信后等待 GPU0 → 通信尾延迟。
  - **计算执行**：GPU0 上 experts 0-1 的 token 处理量远大于 GPU1-3 → GPU0 GEMM 执行时间长 → GPU1-3 完成计算后 idle 等待 GPU0 → 计算尾延迟。
  - **Kernel/硬件层**：论文未明确说明 kernel 细节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoETuner**，一个基于 ILP 的专家放置优化框架，通过两阶段 ILP 求解最优 expert-to-GPU mapping，打破 contiguous block 限制。

  **对应缺陷 1（Token 处理负载不均衡）→ ILP 1: Load-Balanced Expert Clustering**
  - 利用 token routing profiling 收集的 P_{e,l}（每个 expert 的 token 处理量）→ 决策变量 x_{c,e,l}（expert e 是否归入 cluster c）→ 目标 min Σ|T_{c,l} - T̄_l|（最小化 cluster 负载与层均值的偏差）→ 约束每个 cluster 至少一个 expert。
  - **效果**：将高频和低频 expert 混合分配到同一 GPU cluster，使各 GPU 处理的 token 总量接近。例如，layer 14 中 experts 0-1 不再同属一个 GPU，而是与低频 expert 搭配。单节点减少 token 处理尾延迟 36%，平均延迟 34.8%。

  **对应缺陷 2（跨 GPU 通信倾斜）→ ILP 2: Cluster-to-GPU Assignment**
  - 利用跨层 token 路由依赖 R_{e_1,e_2,l} 预计算 cluster 间通信成本 C_{c_1,c_2,l} → 目标 min Σ max(C_{c_1,c_2,l} / B_{g_1,g_2})（最小化每层跨 GPU pair 的通信 tail）→ 约束每个 GPU 等量 expert、cluster-GPU 一对一映射。
  - **核心洞察**：token 在相邻层间存在路由亲和性——若 token 在 layer l 路由到 expert e_1，在 layer l+1 很可能路由到特定少数 expert → 将频繁跨层通信的 expert 放在同一 GPU，消除跨 GPU 通信。
  - **效果**：单节点减少 all-to-all tail 延迟 36.3%，平均延迟 35.4%；多节点减少 tail 30.5%、平均 24.7%。

  全栈执行例子（MoETuner）：
  - **Profiling 阶段**：在 WikiText-103 采样子集上运行 Megatron-LM 推理 → 逐 token 记录路由路径 → 构造 P_{e,l} 和 R_{e_1,e_2,l} 表 → 验证小样本路由统计可近似全数据集模式。
  - **ILP 1 求解**（Gurobi 12.0.0，tolerance 0.025）：输入 P_{e,l}（如 layer 14: expert 0→300 tokens, expert 1→280, expert 2→80, ...）→ 将 8 experts 聚类到 4 clusters → 输出 x_{c,e,l}：cluster 0={expert 0, expert 3}（高+低）、cluster 1={expert 1, expert 6}... → 确保各 cluster token 总量约等于 T̄_l。
  - **ILP 2 求解**：用 x_{c,e,l} 计算 C_{c_1,c_2,l} = Σ R_{e_1,e_2,l} · x_{c_1,e_1,l} · x_{c_2,e_2,l} → 在 B_{g_1,g_2}（NVLink 900GB/s intra-node, IB 400Gbps inter-node）约束下 → 输出 y_{c,g,l}：layer l 中 cluster 0→GPU1, cluster 1→GPU2, ...；layer l+1 中 cluster 0→GPU3, cluster 1→GPU1, ... → 最大化跨层同 GPU expert 对。
  - **部署阶段**：Megatron-LM 加载 Mixtral-8x7B → 读取 expert-to-GPU mapping tensor → 初始化 expert parallel 布局 → 推理时 token dispatch 按优化后 layout 执行 all-to-all → 例如 layer 4 中 expert 7 和 layer 5 中 expert 6 同放 GPU-a → 大比例 token 在层间无需跨 GPU 通信 → 端到端加速 9.3%（单节点 8×H100）和 17.5%（多节点 16×H200）。
  - **Kernel/硬件层**：论文未明确说明 kernel 细节。
