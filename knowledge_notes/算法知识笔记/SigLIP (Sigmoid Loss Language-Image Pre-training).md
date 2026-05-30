## SigLIP (Sigmoid Loss Language-Image Pre-training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SigLIP是Google提出的视觉-语言预训练模型（Zhai et al., 2023），其核心创新是将CLIP的标准softmax对比损失替换为sigmoid loss来处理image-text pairs。与CLIP的softmax loss（需要在整个batch内计算归一化，O(batch²)）不同，SigLIP对每对(image, text)独立使用二分类sigmoid cross-entropy loss，训练更高效且允许任意large batch。此外，sigmoid loss天然支持multi-label匹配（一个image可以匹配多个text），在开放世界理解任务上表现优越。ML-Mamba使用shape-optimized SigLIP（比ViT-Large略大）作为语义编码器。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// SigLIP训练loss vs CLIP loss:
// CLIP: L = -1/N Σ_i log(exp(x_i^T y_i / τ) / Σ_j exp(x_i^T y_j / τ))
// SigLIP: L = -1/N Σ_i log(σ(x_i^T y_i / τ + b) - 1/N² Σ_i Σ_{j≠i} log(σ(-x_i^T y_j / τ + b))
// 其中σ是sigmoid函数, b是可学习的bias, τ是温度参数

// SigLIP在ML-Mamba中的使用:
V_siglip = SigLIP_ViT(X_v)  // ∈ R^{729×D_sig}
// 与DINOv2拼接后送入MSC
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SigLIP开源实现：https://github.com/google-research/big_vision（JAX实现）。HuggingFace上提供PyTorch版本：`google/siglip-base-patch16-256`等。ML-Mamba消融（Table 5）显示SigLIP单独使用在VQAv2上达74.61（优于DINOv2的73.73），在POPE上达87.4（优于DINOv2的86.6），证实SigLIP的语义对齐能力在VLM任务中的核心作用。SigLIP + DINOv2组合成为当前高性能VLM的标准双编码器方案——SigLIP提供language-aligned semantics，DINOv2提供spatial/structural detail。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
