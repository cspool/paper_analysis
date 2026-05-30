## Hierarchical Expert Swap (HierD-ES)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Expert Swap (HierD-ES) 是 HierMoE 提出的分层 expert 交换策略，专门为 HierD-AlltoAll 设计。通过交换两个 expert 在 GPU 间的位置来平衡各 hierarchical group 的通信负载，从而进一步减少 AlltoAll 通信时间。与 SmartMoE 等传统 expert placement 方法不同，HierD-ES 在交换 expert 时统计去重后的 token 分布变化（而非原始 token 数），并根据分层拓扑的带宽差异评估交换对总通信时间的影响。直接计算所有 expert pair 的交换效果需要 O(D·T·K·E²)，通过增量更新方法（四种 case 分析，图 8）降至 O(D·T·K·E)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Input: d* (optimal dimension), routing mask I_route, expert placement P
Output: expert pair (r*, c*) to swap

1. Initialize Z ∈ R^{E×E×U[d*]} ← no-swap dedup token counts per group
   Z_intra ∈ R^{E×E×G} ← no-swap dedup token counts per GPU group

2. For each token t with its K selected experts:
   For each expert pair (A, B) where A is selected, B is not:
     Case analysis (Figure 8 in HierMoE paper):
     - Case 1/2: B's group has no other selected expert
       → B's group token count +1
       - Case 1: A's group has >=2 selected experts → A's group unchanged
       - Case 2: A is the ONLY selected expert in its group → A's group -1
     - Case 3/4: B's group has other selected experts → B's group unchanged
       - Case 3: A's group has >=2 selected experts → A's group unchanged
       - Case 4: A is the ONLY selected expert → A's group -1
   // Incremental update: only adjust affected groups

3. Build estimated time matrix Q_d*[r,c]:
   For each expert pair (r,c):
     n_inter_k ← (U[k]/U[k-1]) · max(Z[r,c,:]) · M · v
     n_intra ← (G/U[d*-1]) · max(Z_intra[r,c,:]) · M · v
     Q_d*[r,c] ← Σ_{i=1}^{d*-1} (n_inter_i · β_inter(i) + α_inter(i))
                  + n_intra · β_intra(d*-1) + α_intra(d*-1)

4. Apply smooth-max (γ=10): smooth_max(x) = max(x) · (Σ (x[i]/max(x))^γ)^(1/γ)

5. (r*, c*) ← argmin Q_d*[r,c]

6. Swap experts r* and c* via NCCL P2P (parameter + optimizer state transfer)
   ≈1% of end-to-end time per swap
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 Megatron-LM 的 expert placement 管理层实现
- 每 iteration（或每 N iterations）执行一次 swap 决策
- HierD-ES 更新频率：每 1 iteration 1.17× speedup vs HD-MoE, 每 8 iterations 1.13×
- 在 DeepSeek-V3 (K=8, E=256) 上：HierD-ES 额外带来 1.13×-1.17× AlltoAll 加速
- smooth-max 参数 γ=10 对性能不敏感（γ∈[5,19] 间加速比 1.16×-1.17×）
- 关键约束：交换后必须重新评估 Z 矩阵（增量更新不保留跨 iteration 状态）

涉及论文标题：
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap
