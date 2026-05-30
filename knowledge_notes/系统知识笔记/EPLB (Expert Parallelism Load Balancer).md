## EPLB (Expert Parallelism Load Balancer)

术语解释
EPLB 是 DeepSeek 提出的 Expert Parallelism 负载均衡器，通过 expert replication 和 placement 策略平衡各 GPU 上的 token 负载。论文 METRO 将其 token routing 策略（均匀分配 tokens 到 replicas 上）作为对比 baseline。

术语是什么？
EPLB 由 DeepSeek 实现并开源（https://github.com/deepseek-ai/EPLB），是当前 MoE 推理系统中广泛采用的 EP 负载均衡方案（被 vLLM、SGLang 等框架集成）。EPLB 在两个步骤中平衡负载：(1) **Expert Replication**: 按上一时间窗口各 expert 处理 token 数的比例创建 replicas——热门 expert 获得更多 replicas，冷门 expert 保持单 replica；(2) **Expert Placement**: 将 replicas 放置在 GPU 上，目标是平衡各 GPU 期望处理的 token 数量。Placement 算法假设后续的 token routing 策略会均匀分配每个 expert 的 token 到其所有 replicas 上。EPLB 的 token routing（第三步）默认将每个 expert 的 token 均匀分配到其所有 replicas。

从系统架构角度拆解术语：
EPLB 在 MoE serving 系统中的执行流程：

```
=== EPLB 三步骤工作流（每时间窗口触发一次）===

Step 1 - Expert Replication:
Input: 上一窗口各 expert 处理的 token 数 stats[1..N], 总 GPU 内存容量
Output: 每个 expert 的 replica 数 R[1..N]
Algorithm:
  total_capacity = G * mem_per_gpu / sizeof(expert)
  for each expert i:
    R[i] = max(1, floor(stats[i] / sum(stats) * total_capacity))
  # 确保 sum(R[i]) <= total_capacity

Step 2 - Expert Placement:
Input: R[1..N], G 个 GPU
Output: placement matrix A[N][G]（expert i 的 replica 放在哪些 GPU）
Algorithm:
  贪心放置：将各 expert 的 replicas 分配到当前 token 负载最低的 GPU 上
  目标: 平衡各 GPU 的期望 token 数

Step 3 - Token Routing (EPLB default):
Input: 当前 batch 中每个 expert i 的 token 数 T[i], placement A
Output: 每个 token 路由到哪个 replica
Algorithm:
  for each expert i:
    evenly distribute T[i] tokens across its R[i] replicas
  # 这使得每个 GPU 上处理的 token 数尽量均衡
```

术语一般如何实现？如何使用？
- EPLB 被 vLLM 和 SGLang 等框架集成，用于 MoE 模型的 EP 部署
- 负载统计基于滑动时间窗口，周期性更新 replicas/placement
- EPLB 的核心假设是 "GPU runtime ∝ 处理 token 数"（compute-bound），但在 memory-bound decode 阶段不成立——METRO 论文实验显示 EPLB routing 在 1.5x replication 下使 activated experts 增加 ~30%，导致 decode latency 退化 14%
- 替代方案（如 METRO）：保留 EPLB 的 replication/placement 策略，但将 token routing 目标从"平衡 token 数"改为"最小化 activated experts 数"
- GitHub: https://github.com/deepseek-ai/EPLB

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---
