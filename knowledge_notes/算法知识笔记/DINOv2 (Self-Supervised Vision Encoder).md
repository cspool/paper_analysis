## DINOv2 (Self-Supervised Vision Encoder)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DINOv2是Meta AI提出的基于自监督学习的视觉基础模型（Oquab et al., 2024），通过knowledge distillation和contrastive learning在LVD-142M（大规模未标注图像数据集）上预训练。核心特点是生成高质量、语义丰富的视觉特征，特别擅长保留低层空间细节（如物体边界、纹理、几何结构）。与CLIP/SigLIP等语言对齐的视觉编码器不同，DINOv2不需要文本监督信号——其训练目标是最小化teacher和student网络输出之间的差异。ML-Mamba使用DINOv2 ViT-Large（304M参数）作为双编码器之一，负责提供图像的空间结构信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// DINOv2在ML-Mamba中的使用:
Input: 图片 X_v ∈ R^{3×384×384}
patches = patchify(X_v, P=14)  // 27×27 = 729 patches
V_dino = DINOv2_ViT_Large(patches)  // ∈ R^{729×D_dino}
// DINOv2内部: 24层ViT transformer blocks with self-attention
// 输出: patch-level dense features

// 与SigLIP特征拼接:
V_siglip = SigLIP_ViT(patches)  // ∈ R^{729×D_sig}
V_img = concat([V_siglip; V_dino], dim=-1)  // ∈ R^{729×(D_sig+D_dino)}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DINOv2开源：https://github.com/facebookresearch/dinov2 (Apache 2.0)。常用加载方式：`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')`。在ML-Mamba消融（Table 5）中，单独使用DINOv2在VQAv2上达73.73，单独SigLIP达74.61，组合达75.26——证明DINOv2的低层空间特征与SigLIP的高层语义特征互补。DINOv2的典型应用场景：(1) 作为多模态模型的空间编码器补充语义编码器（如CLIP/SigLIP）；(2) 密集预测任务（分割、深度估计）；(3) 图像检索和匹配（利用其instance-level特征）。与SigLIP的配合使用时需注意分辨率对齐（两者通常需要统一input size和patch size）。

涉及论文标题：
- ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

---
