## Patch Similarity Entropy (PSE / Patch相似性熵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Patch Similarity Entropy (PSE) 是 PSAQ-ViT (ECCV 2022) 提出的损失函数，用于在 Vision Transformer (ViT) 的 Zero-shot Quantization 中替代 Batch Normalization Statistics (BNS) Loss 作为合成数据生成的先验约束。核心动机：ViT 模型使用 Layer Normalization (LN) 而非 BN，LN 不存储 running mean/variance 等运行时统计信息，因此 CNN 模型常用的 BNS Loss 无法直接应用于 ViT。PSE 的解决思路：利用 ViT 处理图像时 patch token 之间的 similarity 结构作为替代约束——计算合成图像在 ViT 各层产生的 patch token 之间的 cosine similarity 分布（通过 softmax 归一化为概率分布），并最小化该分布的熵与预训练模型在真实数据上计算的参考熵之间的差异。在 Task-Specific ZSQ for Object Detection 论文中，PSE 被用于 Transformer-backbone Mask R-CNN（Swin-T/S）模型的 L_prior。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
def compute_PSE_loss(model, synthetic_batch):
    """计算Patch Similarity Entropy损失"""
    loss = 0.0
    for layer_idx, blk in enumerate(model.blocks):
        # 获取该层的patch token表示
        x = blk.norm1(blk.attn.qkv(synthetic_batch))
        # 计算patch间的cosine similarity
        x_norm = F.normalize(x, p=2, dim=-1)         // (N, num_patches, dim)
        sim = x_norm @ x_norm.transpose(-2, -1)       // (N, P, P) similarity矩阵
        # 计算每个patch的相似性分布熵
        sim_probs = F.softmax(sim / tau, dim=-1)      // tau: temperature
        entropy = -(sim_probs * log(sim_probs + eps)).sum(dim=-1).mean()
        # 与预训练的参考熵对齐
        ref_entropy = model.ref_patch_entropies[layer_idx]
        loss += (entropy - ref_entropy) ** 2
    return loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PSE 在 ZSQ 数据合成阶段替代 BNS Loss，用于所有 LN-based 模型（ViT、Swin Transformer等）。预训练的参考熵值 ref_patch_entropies 需要在真实数据上预先计算一次并保存。在 Task-Specific ZSQ for Object Detection 中，Swin-T/S backbone 的 Mask R-CNN 使用 PSE 作为 L_prior，超参数 {alpha_detect, alpha_PSE, alpha_TV, alpha_l2} = {10.0, 1.0, 0, 1e-3}，合成阶段优化4000次迭代。PSAQ-ViT 和 PSAQ-ViT V2 是关于 PSE 的原始论文。与 BNS Loss 相比，PSE 的优势是不依赖 BN 层，但劣势是需要预先计算参考熵值，且在极低分辨率下 patch 数量少时约束力减弱。

涉及论文标题：
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection
