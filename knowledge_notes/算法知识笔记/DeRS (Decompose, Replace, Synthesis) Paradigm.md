## DeRS (Decompose, Replace, Synthesis) Paradigm

术语解释
DeRS 是由 Huang et al. (2025) 提出的针对 upcycled MoE 模型的参数效率提升范式。核心思想是将 N 个 MoE 专家重构为 1 个专家共享基础权重 + N 个轻量专家专属增量权重，通过消除专家间冗余参数实现极高参数效率。包含三种操作：Decompose（分解为 W_base + Δ_i）、Replace（用稀疏化/量化/低秩矩阵替换 Δ_i）、Synthesis（按需合成 Ŵ_i = W_base + F(Δ_i)）。

术语是什么？
1. **Decompose**：利用 upcycled MoE 专家共享同一初始权重 W_base，将训练后专家分解为 W_i = W_base + Δ_i。余弦相似度 > 0.999 表明 Δ_i 是微小冗余调整。
2. **Replace**：用轻量表示 F(Δ_i) 替换 Δ_i——后处理稀疏化/量化（DeRS Compression）或从训练开始使用稀疏矩阵/低秩矩阵（DeRS Upcycling）。
3. **Synthesis**：推理/训练时按需合成 Ŵ_i = W_base + F(Δ_i)。

从算法pipeline角度拆解术语。
```
def DeRS_forward(x, router, W_base, compact_deltas, k):
    scores = softmax(x @ W_R)
    selected = TopK(scores, k)
    y = 0
    for i in selected:
        Δ_i = reconstruct(compact_deltas[i])  # 稀疏解压/反量化/低秩乘积
        W_i = W_base + Δ_i                    # 加法合成
        y += scores[i] * FFN(x, W_i)
    return y

# Compression: F(Δ_i) = (1-M)⊙Δ_i/(1-p), M~Bernoulli(p) 或 Quant(Δ_i,k)
# Upcycling-SM: F(Δ_i) = torch.scatter(I_i, V_i), I_i固定 V_i训练
# Upcycling-LM: F(Δ_i) = A_i@B_i, [d,r]×[r,d_h]
```

术语一般如何实现？如何使用？
- 稀疏化适合 dense model 经过先验微调（delta 冗余极高，可承受 0.99 drop rate）
- 量化/低秩适合未经过先验微调（需要全局修改能力）
- 仅适用于 upcycled MoE（需共享 W_base），from-scratch MoE 不适用
- 效果：MoE-LLaVA-Phi 上 DeRS-SM 增加 1.11M 参数（2270× 减少）性能 61.1 vs 60.8

涉及论文标题：
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

---
