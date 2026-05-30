## Joint Anisotropy of Query-Key Representations（Query-Key 表示的联合各向异性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Joint Anisotropy of QK 是 Q-Filters 论文（Godey et al., 2025）提出的关于 Transformer 自注意力中 Query 和 Key 表示分布的关键几何发现。该发现整合并深化了两个先前的独立观察：

**先验一（Devoto et al., 2024）**：Key 向量的 L2 范数与平均注意力权重之间存在负相关——低范数 Key 对应高平均注意力，利用此现象可实现 KV Cache 压缩（K-Norm 方法）。

**先验二（Godey et al., 2024）**：Q 和 K 的分布具有各向异性（anisotropic）——它们并非均匀占据 $\mathbb{R}^{d_H}$，而是沿着某个共同方向"漂移"远离原点，且不同头的漂移方向可能相反（$\epsilon = \pm 1$）。

**Q-Filters 论文的核心发现**：上述两个现象可以通过"联合各向异性"统一解释。具体而言：
- **Observation 3.1（联合各向异性）**：存在 $u^h \in \mathbb{S}^{d_H-1}$ 和 $\epsilon = \pm 1$ 使得 $\mathbb{E}(\langle Q_i^h, u^h\rangle) > 0$ 且 $\mathbb{E}(\langle K_j^h, \epsilon u^h\rangle) > 0$。即 Q 和 K 分布在同一个方向 $u^h$ 上有非零均值投影（符号可能相反）。
- **Observation 3.2（单方向性）**：令 $u^h = \arg\max_{u} \mathbb{E}(\langle Q_i^h, u \rangle)$，则对于所有与 $u^h$ 正交的方向 $u_m$（$m \geq 2$），$\mathbb{E}(\langle Q_i^h, u_m \rangle) \approx 0$。即 Q 的各向异性集中在单一方向上。
- **Theorem 3.3（注意力近似）**：$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_j^h \rangle) \approx \kappa^h \langle K_j^h, u^h \rangle$，其中 $\kappa^h = \mathbb{E}(\langle Q_i^h, u^h \rangle) > 0$。

**推论**：在实践中，大多数因果 LM 中 $\epsilon = -1$（Q 和 K 在 $u^h$ 上投影符号相反），因此 $\mathbb{E}(\langle Q_i^h, K_j^h \rangle) \approx -\kappa^h |\mathbb{E}(\cos(K_j^h, u^h))| \cdot ||K_j^h||_2$。这解释了为何 K-Norm（仅用 L2 范数）有效——范数是乘积中的一项，但忽略了角度分量 $\cos(K_j^h, u^h)$。Q-Filters 直接使用 $\langle K_j^h, u^h \rangle$，同时捕捉范数和角度信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**几何验证流程**：

```
// ===== Observation 3.1 验证：联合各向异性 =====
// 输入：预训练模型 M，多个序列 X_1, X_2, ..., X_N
// 输出：主方向 u^h 和各向异性程度的验证

for each head h:
    // Step 1: 收集 Q 和 K 表示
    Q_all = []  // 所有 token 位置的 Q^h
    K_all = []  // 所有 token 位置的 K^h
    for X in [X_1, ..., X_N]:
        Q_all.append(M.forward_get_queries(X, head=h))
        K_all.append(M.forward_get_keys(X, head=h))

    // Step 2: SVD 找主方向
    Q_matrix = stack(Q_all)  // [total_tokens, d_head]
    U, S, Vt = SVD(Q_matrix)
    u_h = Vt[0, :]  // 第一右奇异向量 = 主各向异性方向
    v_2 = Vt[1, :]  // 第二右奇异向量 = 正交方向（用于对比）

    // Step 3: 验证 Observation 3.1
    proj_Q_on_u = Q_matrix @ u_h  // Q 在主方向上的投影
    proj_K_on_u = K_matrix @ u_h  // K 在主方向上的投影

    mean_Q_proj = mean(proj_Q_on_u)  // 应 > 0
    mean_K_proj = mean(proj_K_on_u)  // 可能 < 0（ε = -1）
    // 实践中 ε = -1 在大多数头中成立

    // Step 4: 验证 Observation 3.2（单方向性）
    proj_Q_on_v2 = Q_matrix @ v_2  // Q 在正交方向上的投影
    mean_proj_v2 = mean(proj_Q_on_v2)  // 应 ≈ 0
    // 确认只有 v_1 方向携带各向异性信息

    // Step 5: 验证 Spearman 相关性
    attn_scores = compute_attention_maps(M, X)  // 真实注意力
    S = average_attention_per_position(attn_scores)  // 公式 (2)

    q_filter_scores = K_matrix @ u_h  // Q-Filters 重要性估计
    knorm_scores = ||K_matrix||_2     // K-Norm 重要性估计

    corr_qfilter = spearman_correlation(S, q_filter_scores)
    corr_knorm = spearman_correlation(S, knorm_scores)
    // Q-Filters 相关性 > K-Norm 相关性（大多数头）
```

**几何直观解释**（Figure 3）：
将 $Q^h$ 和 $K^h$ 投影到 SVD 的前两个右奇异向量 $(v_1, v_2)$ 上：K 在 $v_1$ 上的投影颜色编码了该位置的平均注意力——投影值越极端（正向或负向），注意力越高。而在 $v_2$ 上的投影则显示近似零均值的对称分布，与注意力无关。这直观验证了"仅第一主方向编码了注意力选择信息"的结论。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该几何发现的实际用途是构建 Q-Filters：通过 SVD 计算 Query 表示的主方向 $v_1$ 作为 Q-Filter，在推理时用 $\langle K_t^h, v_1^+ \rangle$ 估计 KV pair 重要性。该分析适用于标准 MHA 和 GQA（GQA 需对组内 Q-Filters 取平均），但不适用于使用 QK-normalization 或 attention bias 的模型（因几何特性被修改）。实现不依赖特定框架——只需能提取模型中间激活并执行 SVD（如 NumPy/PyTorch 的 `torch.linalg.svd`）。

涉及论文标题：
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

---
