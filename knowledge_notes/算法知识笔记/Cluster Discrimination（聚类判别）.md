## Cluster Discrimination（聚类判别）

术语是什么？
Cluster Discrimination 是一种自监督表示学习范式：先用 frozen encoder 对大规模数据提取特征，k-means 聚类为语义中心（centroids），再用这些全局聚类中心作为 pseudo-labels 进行多标签语义判别训练。与传统 contrastive learning（CLIP/SigLIP, instance-level + batch-local negatives）和 masked modeling（MAE, pixel reconstruction）不同，Cluster Discrimination 利用全局语义结构而非局部对比。OV-Encoder 扩展到双模态：图像嵌入→2M object-level 类中心，视频嵌入（16帧 concat 特征）→400K motion-level 类中心，合并为 C_uni = C_obj ∪ C_vid（2.4M 类中心）。训练时图像仅对照 C_obj，视频仅对照 C_vid。使用 sigmoid BCE（非 softmax），因一个样本可同时属于多个语义类别（multi-label）。

从算法pipeline角度拆解术语：

```
# 离线聚类（frozen metaCLIP-H14）
e_img = metaCLIP(image)                  # [D] per image
C_obj = kmeans({e_img}, K=2M)            # 2M image centroids
e_vid = metaCLIP(uniform_16_frames)      # [16,D] → concat → [16D]
C_vid = kmeans({e_vid}, K=400K)          # 400K video centroids
per sample: assign top-10 nearest centroids as positive labels

# 在线训练
e = OV_Encoder(sample)                   # ViT + attentive pooling
sim = e @ C_m.T                          # [1, K_m]
loss = sigmoid_BCE(sim, multi_hot_labels)  # multi-label per sample
loss = loss_obj + loss_vid
```

负采样率 r=0.1（仅计算 10% 负类中心），正标签数 l=10，K_obj=2M, K_vid=400K。

术语一般如何实现？如何使用？
离线聚类使用 mini-batch k-means，类中心矩阵作为可学习参数存储于 GPU。适用场景：纯视觉预训练（无语言监督），需要建模细粒度语义结构（intra-class consistency + inter-class relationship）。相比 CLIP-style：优势是全局语义结构，劣势是依赖 frozen encoder 聚类质量。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence
