## MIN-EXP-ROUTING (ILP Formulation for MoE Token Routing)

术语解释
METRO 论文将 Expert Parallelism 中的 token routing 问题形式化为 MIN-EXP-ROUTING —— 一个整数线性规划 (ILP) 问题，目标是最小化各 GPU 上 activated expert replicas 的最大数量。该问题可约化为 classical "scheduling jobs on machines with restrictions to minimize makespan" 优化问题，可通过二分搜索 + bipartite matching (max-flow) 求得最优解。

术语是什么？
MIN-EXP-ROUTING 的形式化定义：
- **输入**: N 个 experts, G 个 GPUs, placement matrix A ∈ {0,1}^{N×G}, token counts per expert T[1..N]
- **决策变量**: x_{i,g} ≥ 0 (expert i 在 GPU g 上处理的 token 数), y_{i,g} ∈ {0,1} (expert i 是否在 GPU g 上激活), λ ≥ 0 (各 GPU 上 activated experts 数的最大值)
- **目标**: min λ
- **约束**: (1) Σ_i y_{i,g} ≤ λ, ∀g (每 GPU activated experts ≤ λ); (2) Σ_g x_{i,g} = T[i], ∀i (所有 token 必须路由); (3) x_{i,g}=y_{i,g}=0 if A_{i,g}=0 (路由遵守 placement); (4) x_{i,g} ≤ T[i]·y_{i,g} (仅激活 expert 可处理 token)
- **Lemma 1**: 任何可行解可约化为"每个 expert 所有 token 路由到单一 replica"而不增加目标值

从编译框架角度拆解术语：
MIN-EXP-ROUTING 的求解策略和复杂度分析：

```
=== 最优解算法（二分搜索 + Bipartite Matching）===

输入: N, G, A, T
输出: y_{i,g}, 最小 λ*

// 二分搜索 λ
low = 1, high = ceil(|A| / G)  // λ 最大为每位 GPU 平均 expert 数
while low < high:
    mid = (low + high) // 2
    
    // 构建 Bipartite Graph B:
    //   Left nodes: experts with T[i] > 0
    //   Right nodes: G GPUs (每个可匹配 mid 次)
    //   Edge (i,g): exists iff A[i][g] = 1
    
    // 可行性测试: 是否存在匹配覆盖所有 left nodes?
    //   将每个 GPU 复制 mid 次 (每个 GPU 可激活 mid 个 experts)
    //   标准 bipartite matching / max-flow
    //   Dinic max-flow complexity: O(|E| * sqrt(|V|))
    //   其中 |V| = N + G*mid, |E| ≤ N*G
    
    if feasible(mid):
        high = mid
    else:
        low = mid + 1

λ* = low
// Total complexity: O(log(|A|/G) × (N+G)² × (|A|/G + N + G))

实测性能 (Qwen3-30B, 8 A100):
  CPU (Dinic max-flow): 116.3μs - 128.8μs (31.4% - 41.3% of FFN time)
  GPU (push-relabel max-flow): 290.0μs - 292.1μs (86.4% - 103.8% of FFN time)
  CPU-GPU data transfer: 26.5μs - 29.2μs (up to 10.4% of FFN time)
  → 最优解开销过大，需要近似算法


=== METRO 贪心近似算法 ===

输入: N, G, A, T
输出: y_{i,g}

// GPU-native 实现 (CUDA kernel, 单 SM)
初始化: L[g] ← 0, lock l_g for each g=1..G
For each expert i = 1..N in parallel:  // 并行度有限 (≤64)
    if T[i] > 0:
        G_i = {g | A[i][g] = 1}         // 候选 GPU 集 (从 placement matrix)
        acquire all locks {l_g | g ∈ G_i} in total order (GPU ID 升序)
        g* = argmin_{g ∈ G_i} L[g]     // 选 activated experts 最少的 GPU
        y[i][g*] = 1
        L[g*] += 1
        release all locks {l_g | g ∈ G_i}

λ = max_g L[g]
x[i][g] = T[i] if y[i][g] = 1 else 0  // 所有 tokens 到单一 replica

复杂度: O(|A|) — 每个 expert-GPU mapping 恰好考虑一次
Routing quality: 在 optimal 的 10.9% 以内
               比 EPLB token-balancing 降低 up to 42.3% activated experts
Kernel 延时: 17μs - 26μs (远远低于 optimal 的 116-292μs)
```

术语一般如何实现？如何使用？
- MIN-EXP-ROUTING 属于 NP-hard makespan minimization with assignment restrictions 问题族，但 N 和 G 在实际 MoE 场景下较小（N=128-256, G=8-16），可通过 ILP or max-flow 求解
- 关键简化：Lemma 1 消除了 x_{i,g}（token 分配）的搜索空间——只需决定每个 expert 在哪个 GPU 激活即可
- 贪心近似有效的原因：(a) 小规模（N≤256, G≤16）限制了 worst-case 近似比；(b) placement matrix A 通常已经过优化（EPLB 的 placement 步骤），给 greedy 留有较好解空间
- 全序锁获取（按 GPU ID 排序）是避免死锁的关键——所有线程获取多个锁时遵循相同顺序
- 实现于单 SM 的原因：并行度受限于 expert 数 + locking 进一步减少并发至 <64，单 A100 SM 足够
- 可用于其他需要 minimizing activated resources 的 EP 负载均衡场景

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens
