## SLO-Aware Resource Pre-allocation for MoE Inference（面向 MoE 推理的 SLO 感知资源预分配）

术语是什么？
SLO-Aware Resource Pre-allocation 是 Remoe 提出的 MMP（Main Model Pre-allocation）算法的核心设计目标：在 serverless 环境中，请求到达时必须立即确定主模型容器的内存规格 w_v，且该规格需要在最坏情况 expert 负载下仍满足 TTFT 和 TPOT SLO。这与传统的在线动态调整策略根本不同——serverless 不支持运行时动态修改已分配资源（会导致冷启动），因此必须在推理开始前一次性预分配足够资源。

MMP 算法的理论基础是 Hoeffding 不等式导出的 expert 负载概率上界：
- **Theorem 1**：n 个 token 通过 layer l 时，第 k 个 expert 处理的 token 数不超过 √(3n)/2 + n/K_l，置信度 95%
- **Corollary 1**：n 个 token 和 m 个 experts 时，总处理 token 数不超过 √(3n)/2 + mn/K_l，置信度 95%

从系统架构角度拆解术语：
MMP 算法（Algorithm 2）流程：
```
1. 初始化 M_min = Σ_l Σ_k (1-x_{l,k}) μ(e_{l,k}) + N_max·D  // 非expert模块最小内存
2. 设 remote expert 比例 b ← 1, M_cal ← m_V^e  // remote experts最小内存
3. repeat:
4.     for l=1 to L:
5.         基于 Corollary 1 和 b 计算每层 remote 处理时间
6.     计算 local experts 所需内存 M^e = f(b)
7.     主模型内存 M ← max(M_min + M^e, M_cal)
8.     用 M 和 b 计算 TTFT 和 TPOT
9.     b ← b - ε  // 逐步降低 remote 比例
10. until TTFT ≤ SLO_TTFT 且 TPOT ≤ SLO_TPOT
11. 选择满足 m_{w_v} ≥ M 的最小内存规格 w_v
12. return w_v
```
关键设计：(1) 使用 worst-case bound 而非平均负载进行预分配，保证 SLO 满足的确定性；(2) 从 b=1（全部 experts remote，最小本地内存）开始递减搜索，找到最小可行 b；(3) MMP 与 SPS 激活预测并行执行——预分配可与 pre-processing 层冷启动重叠。

术语一般如何实现？如何使用？
- 适用前提：需有 MoE 模型各 expert 的 μ(e_{l,k})（内存占用）、τ_{l,k,v}^c（不同 vCPU 规格下的计算时间）的 profiling 数据。
- 实现依赖：Hoeffding 不等式的上界保证需要 expert 激活事件满足独立假设——实际 MoE 中 gating 输出的 token-to-expert assignment 并非严格独立，Theorem 1 提供的是保守上界。
- 与 reactive 方法的对比：传统方法（如 vLLM 的 dynamic batching）在推理过程中动态调整资源，而 MMP 的 pre-allocation 对 serverless 环境必要——一旦容器启动，资源固定。

涉及论文标题：
- Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing
