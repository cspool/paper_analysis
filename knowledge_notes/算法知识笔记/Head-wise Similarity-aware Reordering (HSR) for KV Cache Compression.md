## Head-wise Similarity-aware Reordering (HSR) for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HSR（Head-wise Similarity-aware Reordering）是 ReCalKV 提出的针对 Key 投影矩阵低秩压缩的 attention head 重排序策略。在 grouped SVD 压缩 KV Cache 时，head 的分组方式直接影响近似误差——将具有相似 left singular subspace 的 head 分为一组，可使 SVD 更好地捕获共享子空间结构，从而降低低秩近似误差。HSR 通过三步实现：(1) 计算所有 head 之间的 CKA 相似度矩阵 S ∈ R^{h×h}；(2) 贪心地将 CKA 相似度最高的 head 对分配到同一组（每组大小 s=4）；(3) 剩余 head 填入有空位的组。推理时需对 Key 执行在线 inverse reordering 恢复原始 head 顺序以保证解码等价性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// HSR 流程 (LLaMA-2-7B, h=32 heads, group_size=4)
// Step 1: CKA 计算
W_k ∈ R^{d_model × (h·d_k)}  // Key projection
for i,j in 0..h-1:
    W_i = W_k[:, i*d_k:(i+1)*d_k], W_j = W_k[:, j*d_k:(j+1)*d_k]
    G_i_c = center(W_i @ W_i.T); G_j_c = center(W_j @ W_j.T)
    S[i,j] = Tr(G_i_c @ G_j_c) / sqrt(Tr(G_i_c²)·Tr(G_j_c²))

// Step 2: 贪心分组
groups = [[] * 8]; remaining = set(range(32))
while remaining:
    i,j = argmax_{i,j in remaining} S[i,j]
    assign i,j to non-full group; remaining -= {i,j}

// Step 3: Group SVD + 推理 inverse reordering
order = flatten(groups)
for g in 0..7:
    W_g = concat(W_k heads in group g); L[g],R[g] = SVD_lowrank(W_g, r_g)
// 推理: z_g=x@L[g]; y=z_g@R[g]; inverse_reorder(y) 恢复原始顺序; apply RoPE
```

LLaMA-2-7B, 80% 压缩率：HSR alone 将 WikiText2 PPL 从 9.34 降至 9.01。可视化确认重排序后相邻 head 呈现更高 CKA 相似度。

术语一般如何实现？如何使用？

HSR 完全 offline 执行，PyTorch `torch.linalg.svd()` + 自定义 CKA 计算。group_size=4（32 heads→8 groups）。head permutation 索引需保存用于推理时 inverse 操作，若与 Triton fused kernel 集成则作为 kernel 内在线操作。仅应用于 Key 投影矩阵（Value 投影用 OVC）。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

---
