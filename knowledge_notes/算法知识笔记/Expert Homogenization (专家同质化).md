## Expert Homogenization (专家同质化)

术语解释
Expert Homogenization 是 MoE 训练中的典型退化现象：多个 expert 学到相似特征表示，导致 MoE 退化为近似 dense model，丧失稀疏激活的效率和多样性优势。

术语是什么？
表现: (1) 不同 expert FFN 权重趋同 (cosine similarity 接近 1); (2) router 分配概率接近均匀 (失去区分能力); (3) 随机 shuffle router 分配不影响模型性能。ETR 通过 GrAP 的正交 gating weight 天然防止同质化——每个 w_i 对应 hidden space 的不同扇区，expert 被迫学习其扇区内 token 的专用表征。

同质化的恶性循环:
```
Router随机初始 → 部分expert被过度选择 (rich-get-richer)
→ 被选expert梯度更新更多 → expert能力分化不足
→ router难以区分expert差异 → 继续随机/偏向性路由
→ expert进一步同质化
```

ETR 的打断机制:
```
GrAP正交权重 → 每个expert对应不相交hidden space扇区
→ TCR确保token选最匹配扇区
→ ECR确保expert只处理其扇区内高亲和力token
→ expert被迫在其扇区内专业化 → 正向反馈循环
```

术语一般如何实现？如何使用？
检测: (1) Calinski-Harabasz (CH) Index 测量 expert 间 token 聚类质量 (ETR 使用); (2) expert FFN 权重 pairwise cosine similarity; (3) shuffle router 分配后的性能下降幅度。防止: 正交 router (GrAP)、contrastive loss、mutual distillation loss、expert dropout。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
