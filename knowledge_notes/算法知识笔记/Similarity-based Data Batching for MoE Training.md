## Similarity-based Data Batching for MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Similarity-based Data Batching 是 Lory 用于构造 MoE 训练实例的数据准备策略。标准预训练做法是将随机文档拼接成固定长度实例，可能导致相邻段（segment）来自无关文档。当 segment routing 使用前一段的表示路由当前段时，不相关的相邻段会削弱路由信号的语义一致性，阻碍专家专业化。

Lory 的解决方案：使用 Contriever (Izacard et al., 2022) 计算文档语义相似度，通过贪心搜索将相似文档拼接为训练实例，使相邻段大概率来自相同或相关领域。该方法启发自 In-context Pre-training (Shi et al., 2024, ICLR)，但目标不同——后者旨在增强跨文档边界的推理能力，Lory 旨在促进专家按领域专业化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**数据批处理 Pipeline（论文 Section 3.2, Appendix C）**：

```python
# 输入: 文档集合 D = {d1, d2, ..., dM}
# C: Contriever encoder

# Step 1: 计算文档 embedding
embeddings = {d: C(d) for d in D}

# Step 2: 构建相似度图（使用 FAISS 近似搜索）
N = {}  # adjacency
for d in D:
    topk = FAISS.search(embeddings[d], k=top_k)  # top-k most similar
    N[d] = {d_j: cosine_sim(embeddings[d], embeddings[d_j]) for d_j in topk}

# Step 3: Greedy concatenation (Shi et al., 2024 的算法)
instances = []
remaining = set(D)
while remaining:
    current = random.choice(list(remaining))
    instance = [current]
    remaining.remove(current)
    while instance_length(instance) < L:  # L = 4096 tokens
        candidates = [d for d in N[current] if d in remaining]
        if not candidates:
            break
        # 选择与 current 相似度最高且未被使用的文档
        next_doc = argmax(candidates, key=lambda d: N[current][d])
        instance.append(next_doc)
        remaining.remove(next_doc)
        current = next_doc
    instances.append(concat(instance))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键：
- **Contriever**：无监督对比学习训练的 dense retriever (Izacard et al., TMLR 2022)，用于编码文档为固定维度向量。Lory 使用预训练 Contriever 不做微调。
- **FAISS (Johnson et al., 2019)**：Facebook 的高效近似最近邻搜索库，Lory 使用 FAISS GPU 版本进行十亿级文档的相似度搜索。
- **贪心搜索**：从随机文档开始，每次选择与当前文档最相似且未被使用的文档追加到实例，直到实例达到 token 预算（4096）。如果当前文档没有可用相似文档，重启新实例。
- **与 Standard Random Batching 的对比**：Random batching 将随机文档拼接，相邻段可能来自无关领域（如医学论文 + 餐厅评论）。Similarity-based batching 确保相邻段语义相关，提供更一致的路由信号。
- **对 MoE 训练的具体影响**：Similarity batching 下 Lory MoE 相对 Dense 的 loss 改善显著大于 random batching（图 4 right），且差异随训练数据量增加而放大。
- **性能开销**：Contriever encoding + similarity graph construction 是一次性预处理，不影响训练 throughput。FAISS 搜索和贪心拼接在数据预处理阶段完成。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training
