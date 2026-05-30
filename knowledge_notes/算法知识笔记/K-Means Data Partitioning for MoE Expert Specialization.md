## K-Means Data Partitioning for MoE Expert Specialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE-DisCo 提出的基于无监督聚类的训练数据划分方法，将原始数据划分为 E 个语义区分的子集，各分配给一个 expert 以促进专业化。流程：(1) 预训练 embedding 层编码句子所有 token，mean pooling 得到固定维度句子向量 h_x；(2) K-Means（K=E）聚类最小化簇内平方距离和；(3) 每个簇映射为一个数据子集 D_k。目标函数：min Σ_{k=1}^{K} Σ_{h_x∈C_k} ||h_x - μ_k||²。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# K-Means 数据划分（离线，一次性）
for sentence x in D:
    for token x_i in x:
        e_i = Embedding(x_i)              # token embedding [d_embed]
    h_x = (1/n) · Σ e_i                   # mean pooling [d_embed]

{C_1, ..., C_E} = KMeans({h_x}, K=E)     # 聚类
for k in 1..E:
    D_k = {x | h_x ∈ C_k}                # 簇→数据子集映射
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
聚类在训练前离线执行一次。消融实验（Figure 6）验证：随机分配替代 K-Means 后 fine-tune 性能退化至 Full-Parameter 水平，证明语义区分的分配对 expert 专业化至关重要。该方法与 LRP（Latent Prototype Routing）和 domain-adaptive pre-training 的动机一致——通过无监督发现数据内在结构指导训练。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models
