## Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll) 是 HierMoE 提出的分层 token 去重 AlltoAll 算法，用于减少 MoE 训练中 Expert Parallelism 下的通信冗余。核心原理：在 GPU 集群的分层拓扑结构（如 4 层：Inter-Node/IB → Inter-QPI → Inter-NVLink → Intra-NVLink）中，不同层级的 AlltoAll 操作将 experts 划分为不同大小的 group（如 Inter-Node 按 4 nodes 分为 4 groups，Intra-GPU 按 32 GPUs 分为 32 groups）。当 top-K 中多个 expert 位于同一 group 时（如 K=8, R=4 → 55% 重复率），同一 token 在 AlltoAll 中被冗余传输。HierD-AlltoAll 在每层 AlltoAll 前执行 token 去重：将 routing mask I_route ∈ R^{T×E} 按 expert group 聚合为 I_route ∈ R^{T×U[i]}，同一 group 内多 expert 选中 → 仅传输一份 token 副本。通过线性性能模型 t_d = Σ(Inter-level time) + Intra-level time（公式 3），自动选择使总通信时间最小的维度 d*。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HierD-AlltoAll 在 MoE layer 的 dispatch 阶段执行：

```
Input: I_route ∈ R^{T×E} (boolean routing mask), M (embedding dim),
       G (GPUs), E (experts), D (topology levels), U[D] (group sizes per level)
Output: Optimal d*, executing token dispatch with deduplication

Step 1 -- Compute HD1 (standard AlltoAll) time:
  m ← E/G                              // experts per GPU
  for j=1..G:
    I_route_g[i,j] ← OR_{j1=(j-1)m+1}^{j·m} I_route[i,j1]
  p[j] ← Σ_i I_route_g[i,j]            // dedup token count per GPU group
  n_a2a ← G · max(p) · M · v           // total comm bytes (v=2 for FP16)
  t1 ← α + n_a2a · β

Step 2 -- Compute HDd for d=2..D:
  for k=1..d-1:                         // Inter-level layers
    m ← E/U[k]
    I_route_k[i,j] ← OR_{j1=(j-1)m+1}^{j·m} I_route[i,j1]
    p_k[j] ← Σ_i I_route_k[i,j]        // dedup tokens in U[k] expert groups
    n_inter_k ← (U[k]/U[k-1]) · max(p_k) · M · v
    Update I_route to reflect post-Inter-level-k token distribution
  n_intra ← (G/U[d-1]) · max(p_d) · M · v
  td ← Σ_{i=1}^{d-1} (n_inter_i · β_inter(i) + α_inter(i))
       + n_intra · β_intra(d-1) + α_intra(d-1)

Step 3 -- Select optimal dimension:
  d* ← argmin_{1≤d≤D} td
  Complexity: O(D·T·K) where T=tokens, K=top-K
```

关键权衡：高层（小 group 数 → 大去重收益，如 R=4 时 K=8 重复率 55%）+ 低带宽链路（IB 200Gb/s）→ 去重后大幅减少 Inter-node 通信量。低层（大 group 数 → 小去重收益）+ 高带宽链路（NVLink 112.5GB/s）→ 去重收益有限但带宽充裕。d* 自动权衡这两者。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现于 Megatron-LM 的 MoE token dispatcher 中，替换标准 NCCL AlltoAll
- 集群启动时用 nccl-tests 一次性测量 7 种 AlltoAll 变体的 α, β 参数（r² > 0.997, <300s 测量 + <10ms 拟合）
- 每 iteration 在 CPU 控制逻辑上计算 d* (O(D·T·K)，微秒级)
- HierMoE 在 32-GPU 集群上实现 vs Megatron-LM AlltoAll 1.99×-2.72× 加速

涉及论文标题：
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap
