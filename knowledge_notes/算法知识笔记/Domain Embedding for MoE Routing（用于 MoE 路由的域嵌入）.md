## Domain Embedding for MoE Routing（用于 MoE 路由的域嵌入）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Domain Embedding for MoE Routing 是 Nexus 提出的用预计算的域级语义嵌入替代随机初始化作为 MoE router 输入的技术。传统 MoE router 用随机初始化的线性层参数 W_r 处理每个 token，与域语义无关；Nexus 则用外部 embedding model（论文使用 Cohere Embed v3）对每个域对应的训练数据集编码，将编码向量平均得到 d_i ∈ R^m 作为该域的"域嵌入"，然后通过投影层 P_r 将 d_i 映射为该域 expert 的 expert embedding e_i。域嵌入在训练前一次计算并存储，训练和推理期间不更新。替代方案：如果使用无监督聚类划分域（如 c-BTM 的 Gururangan et al. 2023），centroid 可代替 embedding model。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 域嵌入预计算（离线，一次完成）
for domain_i in [ArXiv, Books, C4, SE, Wiki]:
    domain_data = load_domain_data(domain_i)
    embeddings = []
    for doc in domain_data:
        emb = embed_model.encode(doc)        # [m]
        embeddings.append(emb)
    d_i = mean(embeddings, dim=0)            # [m]

# 投影后 expert embedding 的域间关系（Nexus Figure 8）:
# 投影前 cosine similarity: Books-C4 ≈ 0.6, GitHub-SE ≈ 0.7
# 投影后: 相对关系保持但整体 pushed apart（lower inter-expert similarity）
# P_r 的学习目标: 在保持域关系的条件下增大 expert embedding 间距
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：(a) 外部 embedding model：需要与 LLM 独立的 embedding service；(b) 预计算：对每个域的整个训练集编码 → 平均 → 存储为 [m, n_domains] 张量；(c) 在 router 训练中作为固定输入（不参与梯度更新）。
- **适用条件**：需要预先划分的数据域结构（如 SlimPajama 的 sub-dataset）；域嵌入质量依赖 embedding model 的表征能力。
- **灵活性**：P_r 投影保持域间相对关系（Figure 8），使得语义相近的域（Books & C4）的 expert embedding 也相近——token 可能被交叉路由——这一特性实现了隐式的跨域知识共享，同时避免了同一 token 被完全不相关的 expert 处理。

涉及论文标题：
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

---
