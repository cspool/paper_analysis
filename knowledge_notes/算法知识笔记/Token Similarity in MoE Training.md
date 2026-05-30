## Token Similarity in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Similarity（Token相似性）是指在MoE训练的all-to-all通信中，发送到同一个expert的token在hidden representation空间中呈现高度相似的现象。LSH-MoE论文通过Principal Component Analysis (PCA)降维可视化发现，输入到all-to-all通信的token形成明显的聚类结构（clustering phenomenon）。

Token相似性的来源被归因为两个主要因素：
1. 数据因素：真实世界数据遵循Zipf's Law，导致某些数据元素比其他元素更频繁出现，形成token表示的偏斜分布。
2. 模型结构因素：Transformer的attention机制会捕获和整合token间的上下文信息，从而在句子级别均质化（homogenize）token表示，增强共享语义关系。

这一观察是LSH-MoE方法的核心动机：因为token高度相似，所以可以用聚类中心替代完整token进行all-to-all传输，且仅损失少量信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Token相似性在MoE pipeline中的表现：

```
# 在MoE layer中，gate网络将token分配给不同expert
# 分配到同一expert的token集合X_i存在内在相似性

# 可视化token分布（论文Figure 4）:
# 对all-to-all通信中的token做PCA降维到2D
# 结果：token自然聚类成若干个group
# 每个cluster内的token在语义上相似（如共享类似的上下文）

# 利用token相似性压缩通信:
# 不传输所有N个token，而是:
# 1) 将N个token聚成m个cluster (m << N)
# 2) 仅传输m个cluster center
# 3) 接收端用center + 保存的残差恢复近似token
# 压缩率 = m/N ≈ 20% (6个hash函数时)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Token相似性不需要显式实现——它是被观测到的数据特征
- 利用方式：使用LSH等快速聚类方法将相似token分组，以group-level信息替代instance-level信息
- 在实践中，token相似性不仅存在于NLP模型（RoBERTa, GPT, T5），也存在于CV模型（Swin-MoE），说明这是MoE架构中通信数据的普遍特性
- 论文通过PCA可视化提供了token相似性的实验证据，但不需要在实际训练中对token做PCA分析——直接使用LSH聚类即可隐式利用此特性

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
