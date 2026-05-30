## Pairwise Expert Similarity (PES)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pairwise Expert Similarity (PES) 是 SIMBAL 论文提出的轻量级 expert 冗余度量指标。定义为所有 expert 输出之间成对余弦相似度的 batch 平均值：

C_expert(x) = (2/(N(N-1))) * Σ_i Σ_{j>i} cos(f_i(x), f_j(x))

PES = (1/|B|) Σ_{x∈B} C_expert(x)

其中 N 为 expert 数，f_i(x) 为 expert i 对输入 x 的输出向量，cos(u,v) = u·v/(||u||·||v||)。PES 越低表示 expert 输出越多样化（越不冗余），expert 专精化程度越高。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PES 计算流程：

for each batch sample x:
    outputs = []  # compute all N expert outputs
    for expert_id in range(N):
        h = silu(x @ W_gate[i]) * (x @ W_up[i])
        out = h @ W_down[i]
        outputs.append(out)
    similarities = []
    for i in range(N):
        for j in range(i+1, N):
            sim = cosine_similarity(outputs[i], outputs[j])
            similarities.append(sim)
    C_expert = mean(similarities)
PES = mean(C_expert over batch)

计算开销：需要在每个 expert 上做一次 forward pass → FLOPs 约为 full model 的 3.6-4.9x（但仅对验证集子集做一次），远少于 dropout-based 评估需要数百次 full model 验证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **vs Dropout-based 评估**（Dai et al. 2024 的 expert dropout 方法）：dropout 方法需对每种 expert 组合做 full validation → 计算量巨大且缺乏粒度。PES 只需一次 full expert inference → 可做 per-layer、per-checkpoint 的细粒度分析
- **使用方式**：训练过程中定期计算 PES on validation set（4M tokens），监控 expert 冗余度变化
- **SIMBAL 结果**：MoE-L 上 SIMBAL min PES = 0.0028 vs LBL 0.0241（约 8.6x 更低），表明 SIMBAL 训练的 expert 专精化程度显著更高
- **局限性**：需对所有 expert 做 forward → 仅适用于离线分析，不能作为在线训练指标；对个别 layer 的 outlier spikes 敏感

涉及论文标题：
- Load Balancing Mixture of Experts with Similarity Preserving Routers
