## Token Saliency in Video DiT Attention (视频 DiT 注意力中的 Token 显著性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Saliency 在 video DiT attention 中指某个 token 在 attention 分布中接收到的总注意力权重。形式化定义：给定 attention map A = softmax(Q·K^T/√d_k) ∈ R^(h×L×L)（h=heads, L=sequence length），token j 的 saliency 定义为 s_j = Σ_h Σ_i A[h,i,j]，即所有 query 对该 token 的注意力之和。QuantSparse 论文在 empirical analysis 中发现 video DiT 的 token saliency 分布呈**重尾分布（heavy-tailed distribution）**：仅 <10% 的 tokens 占据大部分 attention mass（见图 3a, 6, 7——Wan2.1 和 HunyuanVideo 的多个 block 普遍呈现此现象）。这一现象源于 video data 的时空局部性：相邻 spatial/temporal tokens 高度相似，attention 自然集中到少数关键 tokens。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Token saliency 在 MSAD 的 Local Guidance 中的使用：

```
// Compute FP attention
A_fp = softmax(Q_fp·K_fp^T / √d_k) ∈ R^{h×L×L}

// Token saliency (Eq. 7)
for j in 0..L-1:
    s_j = 0
    for h in 0..H-1:
        for i in 0..L-1:
            s_j += A_fp[h, i, j]   // aggregate attention received by token j

// Select top-k salient tokens
I = argsort(s, descending=True)[:k]  // k=256, <2.5% of L≈10⁴

// Local distillation: only on salient queries
A_local = softmax(Q_q[I,:]·K_q^T / √d_k)   // ∈ R^{k×L}
L_local = MSE(A_fp[I,:] || A_local)
```

效果对比：salient selection (PSNR 16.82) vs random selection (PSNR 15.49), 说明 top-k salient 选择显著优于随机采样。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Token saliency 在 QuantSparse 校准阶段每 block 计算一次（因 FP 前向固定, saliency 不变），后续优化迭代中复用 I 索引（无需重复计算）。这一特性与 video data 的 temporal coherence 相关——关键 spatial-temporal tokens 在 denoising 过程中保持稳定。具体选择 k=256（平衡效果与效率），在 s=128 的 global guidance 基础上提供重要的 local 补充。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---
