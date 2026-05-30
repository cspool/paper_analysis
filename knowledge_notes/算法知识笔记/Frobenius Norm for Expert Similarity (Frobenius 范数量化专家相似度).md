## Frobenius Norm for Expert Similarity (Frobenius 范数量化专家相似度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frobenius Norm（Frobenius 范数）在 MoLA 中被创新性地用作量化 MoE 架构中不同 expert 之间差异（等价于相似度/冗余度）的度量工具。定义为：$\|A - B\|_F = \sqrt{\sum_{i,j} |a_{ij} - b_{ij}|^2}$，即逐元素差的平方和的平方根。MoLA 计算每层所有 expert pair（任意两个不同 expert 的合并 LoRA 权重矩阵 B_e@A_e）的 Frobenius Norm 均值，值越小 → expert 越相似 → 冗余越高。

从算法pipeline角度拆解术语：
```
# 输入: 每层 j 所有 expert 的 LoRA 矩阵 {A_e, B_e}_{e=1..N_j}
# 输出: 每层的 mean Frobenius Norm

for layer_j in 1..m:
    W_list = []  # expert 等效权重
    for e in 1..N_j:
        W_list.append(B[e] @ A[e])  # [d_p, r] @ [r, d_q] → [d_p, d_q]
    
    norms = []
    for p in 1..N_j:
        for q in p+1..N_j:
            diff = W_list[p] - W_list[q]            # [d_p, d_q]
            fn = sqrt(sum(diff ** 2))               # Frobenius Norm
            norms.append(fn)
    
    layer_mean_fn[layer_j] = mean(norms)
    # 值越大 → expert 越多样化 → 该层越受益于更多 expert
```

MoLA 关键数值发现（LLaMA-2-7B, 32 层, MoLA-□ 8888, instruction-tuned）：
- Layers 1-8: mean FN ~0.1-0.2 → 高冗余 → 可减少 expert
- Layers 9-24: mean FN ~0.3-0.4 → 中等
- Layers 25-32: mean FN ~0.5-0.6 → 低冗余（专家差异化大）→ 应分配更多 expert

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch: `torch.norm(W_p - W_q, p='fro')` 或手动 `torch.sqrt(torch.sum((W_p - W_q) ** 2))`
- 适用场景：(1) MoE expert redundancy 定量分析；(2) 指导 expert pruning/allocation 决策；(3) 与其他相似度量（Cosine Similarity, CKA）互补。
- 注意事项：Frobenius Norm 受矩阵 scale 影响——MoLA 中所有 expert 从相同初始化开始（A: randn, B: zeros），scale 差异天然反映功能分化程度，因此适用。若 expert 经历不同训练动态导致 scale 差异，建议同时使用 Cosine Similarity。

涉及论文标题：
- MoLA: MoE LoRA with Layer-wise Expert Allocation
