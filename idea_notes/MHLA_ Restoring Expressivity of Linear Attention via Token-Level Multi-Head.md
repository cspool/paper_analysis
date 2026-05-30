## MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

- baseline方法是什么？
  - **Softmax Self-Attention**：对每个 query q_i，计算与所有 key k_j 的 pairwise 相似度 exp(q_i·k_j/√d)，对所有 value v_j 加权求和。复杂度 O(N²d)，内存 O(N²)。全栈执行例子：输入 token 序列 X → 线性投影 Q,K,V → 计算 QK^T/√d → softmax 归一化 → ×V 输出 → 结果传入下游 FFN/layer norm。当 N 增长时，QK^T 矩阵占据 O(N²) 内存，成为瓶颈。
  - **Linear Attention**：将 softmax kernel 替换为可结合的 feature map φ(·)，使得 Sim(Q_i, K_j) ≈ φ(Q_i)φ(K_j)^T。通过先计算全局 KV summary G = Σ_j φ(K_j)^T V_j ∈ R^(d×d)，查询只需 q̃^T G / q̃^T z。复杂度 O(Nd²)，内存 O(d²)。全栈执行例子：输入 X → 投影 Q,K,V → 应用 φ 得 Q̃, K̃ → 计算全局 summary G = Σ_j K̃_j^T V_j 和 normalizer z = Σ_j K̃_j → 对每个 q̃_i 计算 o_i = (q̃_i^T G) / (q̃_i^T z)。**缺陷**：所有 token 被压缩进同一个 d×d 的全局 summary，rank ≤ d（通常 d_h ≤ 72），导致"全局上下文坍缩"——当 N >> d 时注意力矩阵 rank 不足、熵升高（分布趋于均匀），失去 query-conditioned 的选择性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MHLA (Multi-Head Linear Attention)**：沿 token 维度将序列分为 M 个 non-overlapping blocks，为每个 block 独立计算局部 KV summary，再通过可学习的系数矩阵 Mc 使每个 query block 生成专属的混合 summary。全栈执行例子：
    1. 输入 X ∈ R^(N×d) → 投影 Q,K,V → 应用 φ 得 Q̃, K̃
    2. 沿 spatial/spatiotemporal 维度将 N 个 token 划分为 M 个 blocks
    3. 每 block b 独立计算局部 summary：S_b = Σ_{j∈b} K̃_j^T V_j ∈ R^(d×d)
    4. 通过可学习系数矩阵 Mc ∈ R^(M×M)（初始化为 locality-biased：m_{i,j}^(0) ∝ 1-dist(i,j)/max_dist），query block i 混合所有 summary：S̃_i = Σ_b m_{i,b} S_b
    5. 输出：o = (q̃^T S̃_i) / (q̃^T z̃_i)，其中 token 级别贡献为 m_{i,b(t)} (q̃^T K̃_t) V_t^T
    6. 结果传入下游 FFN/layer norm → 下一层
  - **解决 Baseline 缺陷的对应关系**：
    - **Rank 限制**（基线：d → MHLA：Σ_b min(N_b, d)）：将单个 d×d summary 拆成 M 个局部 summary，再通过学习混合恢复多样性。当各 block 的 row spaces 线性独立时，rank 可接近 Σ_b min(N_b, d)，远超 d。实测 MHLA 的 attention score rank 显著高于所有 linear attention 变体（Fig. 3b）。
    - **注意力熵过高/稀疏性丧失**（基线：uniform distribution → MHLA：concentrated distribution）：Mc 允许每个 query block 选择性关注相关 blocks（block-level pruning），block 内部再通过 q̃^T K̃_t 区分 token 贡献（token-level reweighting），两阶段机制恢复 query-conditioned 的尖锐注意力分布。实测 MHLA 熵低于 linear attention 甚至 softmax attention（Fig. 3b）。
    - **不引入额外模块**：与 Focused LA（加 DW-Conv）、Inline Attn（加卷积+gating）等不同，MHLA 仅需标准 GEMM 操作和可学习系数矩阵，额外开销 O(M²d²)。当 M² ≤ N 时，主导项仍为 O(Nd²)。
  - **对比 baseline 的关键差异**：
    - **Linear Attention (单个全局 summary)** → **MHLA (M 个局部 summary + 可学习混合)**：Linear Attention 对所有 query 共享同一个 G，导致 query 间无差异（rank ≤ d, entropy 高）。MHLA 每个 query block 通过专有系数 m_i 混合各 block summary，恢复 query-conditioned 选择性。ImageNet DeiT-T: Linear Attn 69.8% → MHLA 75.8%（+6.0%）；DiT-S/2 FID: Linear Attn 89.72 → MHLA 59.80（↓33%）；Wan2.1 视频生成: Linear Attn Total 58.24 → MHLA 82.62（+41%）。
    - **Self-Attention（O(N²) pairwise）** → **MHLA（O(N) block-level + token reweighting）**：Self-Attention 通过 pairwise softmax 实现完全 query-conditioned 但 O(N²) 复杂度。MHLA 通过两阶段（block 选择 × token 重加权）逼近相同效果。DiT-XL/2 256px FID: Self Attn 19.47 → MHLA 20.32（w/o CFG 相近），512px 下 MHLA 吞吐量是 Self Attn 的 2×；视频生成 31500 tokens: MHLA latency 81s vs Wan2.1-FA 166s（2.1× speedup）。
