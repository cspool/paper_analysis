## PRISM 调度算法（Multiple-Choice Knapsack / 动态规划 batch 形成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PRISM 是 RESONATOR 的 Inter-GPU Parallelism Engine 核心调度算法：给定 pending encoder 请求队列 Q={R_1,...,R_m} 与可用 GPU 数 N，把"如何组 batch + 给每个请求选 DP/TP 度"建模为 Multiple-Choice Knapsack Problem（MCKP）。对每个请求 R_i，合法 TP 集 K_i⊆{1,2,4,...,N}（显存过滤），调度价值 v(R_i,k)=1/T(R_i,k)（处理速率=延迟倒数，最大化单位时间完成的工作量即吞吐），成本=消耗 GPU 数 k。优化目标：max Σ_i v(R_i,k_i) 且 Σ_i k_i≤N、k_i∈K_i∪{0}（0=不调度）。用 DP 求解：dp[i][j]=max(dp[i-1][j], max_{k∈K_i,k≤j}{dp[i-1][j-k]+v(R_i,k)})，回溯 trace 表重构最优 batch 与各请求 TP 度。动机：请求异构造成 Head-of-Line（高分辨率长请求阻塞短请求），且 latency-concurrency 权衡（TP 降单请求延迟但占更多 GPU、减并发；DP 反之）随系统负载动态变化，静态并行度必然次优。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PRISM 的调度流程（Algorithm 2）：
```
输入: Q=[R_1..R_m], GPU 数 N, Atlas Λ
dp[m+1][N+1] ← 0; trace[m+1][N+1] ← 0
for i in 1..m:
  for j in 1..N:
    dp[i][j] ← dp[i-1][j]; trace[i][j] ← 0        # 默认跳过 R_i
    for k in Λ.ValidTP(R_i) 且 k≤j:
      v ← 1/Λ.GetLatency(R_i,k)
      if dp[i-1][j-k]+v > dp[i][j]: dp[i][j] ← dp[i-1][j-k]+v; trace[i][j] ← k
# 回溯
B_opt ← ∅; j ← N
for i in m..1:
  k ← trace[i][j]
  if k>0: B_opt ← B_opt ∪ {(R_i, TP=k)}; j ← j-k
return B_opt    # dp[m][N] = 最优聚合吞吐
```
Annotations：价值用 1/延迟 使目标天然偏向"单位时间完成更多请求"；DP 保证全局最优（每个请求至多选一个 TP 选项=MCKP 的 multiple-choice 约束）；调度结果交给 Unified Runtime 经 logical sharding 执行（零开销切换使其可行）。案例：20 请求混合 4 种分辨率时，动态调度把高分辨率路由到 latency-optimal TP、低分辨率到 throughput-optimal DP，比最差静态 8TP 快 3.23×、接近 Oracle。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：标准 0/1 背包变体的二维 DP + 回溯（O(m·N·|K|)）；在线每次 batch 形成时运行，查询 Performance Atlas 得到各 (R_i,k) 的延迟。使用：请求到达排队 → 周期性/触发式对队列跑 PRISM → 得到本批请求与各自 TP 度 → logical sharding 以控制面 ld 参数更新执行。它是把组合优化（MCKP/DP）作为 serving 调度器的代表用法——与 priority/queueing 式调度器不同，PRISM 显式建模"GPU 预算约束 + 每请求多可选并行度"。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
