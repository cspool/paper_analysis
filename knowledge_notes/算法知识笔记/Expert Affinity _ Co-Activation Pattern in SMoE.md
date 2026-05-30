## Expert Affinity / Co-Activation Pattern in SMoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Affinity（专家亲和力）是 Sparse MoE 中两个 expert 被同一 token 同时激活的频率。在 top-k routing 下，每个 token 每层选择 k 个 expert。Affinity matrix $A \in \mathbb{R}^{n \times n}$ 记录所有 token 上 expert pair (i, j) 的共激活次数。GRACE-MoE 在 offline profiling 阶段通过 calibration data 构建 per-layer affinity matrix，发现 expert 间存在强 co-activation pattern——某些 expert 几乎总是一起被选中（处理特定领域知识），另一些几乎从不同时出现。C2R (Zhang et al. 2025a) 也独立发现此模式用于 collaboration-constrained routing。GRACE-MoE 将其作为 spectral clustering 的输入来指导 expert grouping——高 affinity expert 对放在同 GPU/同节点以减少跨设备 All-to-All 通信。Cross-dataset transfer 实验表明 affinity pattern 在数据集间稳定（最差 latency 增加 ≤4.52%），意味着 offline placement 可跨 dataset 复用。

从算法pipeline角度拆解术语：

```
# Profiling 构建 affinity matrix (per layer)
A[l] = zeros(n_experts, n_experts)
for token t in calibration_data:
    topk = router[l](h_t)  # k=6 or 8
    for i in topk:
        for j in topk:
            if i != j: A[l][i][j] += 1

# Affinity 指导 grouping:
# 高 A[i][j] → 同组 → 减少跨设备通信
# 低 A[i][j] → 可分到不同组 → 保持负载灵活
# Cross-node: fully non-uniform grouping (无 size 约束)
# Intra-node: controlled non-uniform (ratio r 约束 size deviation)
C = SpectralClustering(A, D)
```

术语一般如何实现？如何使用？

- Calibration data 通常数千到数万 token，从训练/验证集采样
- Affinity matrix 可直接作为 spectral clustering 的 weighted adjacency（无需额外归一化）
- 混合 dataset profiling 获得最鲁棒的 affinity estimation
- C2R 用 affinity 限制 routing（token 只能选组内 expert），GRACE-MoE 用 affinity 指导 placement 且保持 routing 不变（lossless）

### Expert-Expert Collaboration (ECC) as Dropping Criterion (Jaiswal et al. 2025)

MC-Suite 中的 ECC (Expert-Expert Collaboration) 准则从 pruning 视角利用 co-activation 模式：给定 calibration data，定义 collaboration matrix C_{p,q} = Σ 1[K_i ∩ {E_p, E_q} == {E_p, E_q}]（两 expert 共同被路由到同一 token 的次数）。高 collaboration 的 expert pair → 一个可被丢弃（因为另一个可覆盖相同 token 的处理任务）。具体丢弃决策：从 collaboration matrix 中选 min/max 值的 expert pair，再结合 EUF（Expert Usage Frequency）选择使用频率更低的那个丢弃。这与 GRACE-MoE/HD-MoE 用 affinity 做 placement grouping 不同——ECC 将 co-activation 信息用于压缩而非通信优化。

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

### HD-MoE 中的 Co-Activation
HD-MoE 独立发现并利用了 expert co-activation 模式（图 3c 的 Expert Routing Affinity heatmap，值 (i,j) 表示给定 expert i 被激活时 expert j 也被激活的条件概率）。HD-MoE 将此模式用于通信模型：定义 expert group g，f_g 为 group co-activation 频率，t̂_comm = (4/BW)·max_c{ Σ_g (Π_{i∈g} ⌈P_ic⌉)·f_g·B·h }。Co-activation 量化使 LP 能准确估计不同 placement 方案的通信开销。
