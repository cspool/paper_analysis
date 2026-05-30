## Domain-level Expert Specialization with Segment Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Domain-level Expert Specialization 是 Lory 训练后观察到的专家行为模式：使用 segment-level routing 训练的 MoE 模型，其专家自动学习按领域（domain）而非按词法特征（token-level features）进行专业化。具体表现为：不同层级的专家对特定领域（arXiv 学术论文、Python 代码、Books、Wikipedia）展现出不同的路由偏好权重（图 6），且低层专家的领域偏好较平坦，中高层的领域分化更明显。

与 token-level MoE routing（如 Switch Transformer, Expert Choice）的对比：
- Token-level MoE 专家学到的是浅层特征：某些专家专门处理标点、冠词、介词，某些专门处理数字、动词等（Zoph et al., 2022; Jiang et al., 2024; Xue et al., 2024）
- Segment-level MoE 专家学到的是深层语义领域特征：专家按主题/领域分化，在训练数据（CommonCrawl）中低频的领域（如 Python code）也能获得专业化专家

这种领域级专业化的驱动力来自 similarity-based data batching：通过将语义相似的文档拼接为训练实例，相邻段来自同一领域，段级路由因此学习到领域感知的路由策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**专家专业化分析流程（论文 Section 5.4）**：

```python
# 输入: 0.3B/8E 模型，来自 4 个领域的评估数据（Books, arXiv, Python, Wikipedia）
# 对每层 l 和每个领域的数据 D_domain：
for domain in [Books, arXiv, Python, Wikipedia]:
    for segment in domain_data:
        h_bar = segment.mean_repr              # segment avg hidden state
        e = softmax(R(h_bar))                  # routing weights (E-dim)
        routing_weights[l][domain] += e        # accumulate
    routing_weights[l][domain] /= len(domain_data)  # average

# 可视化: 热力图 (expert × domain) 每层一个
# 观察:
# - Layer 0: routing weights flat across domains (所有专家权重相似)
# - Layer 11: clear domain specialization (expert 7 for arXiv, etc.)
# - Layer 23: clear domain specialization (distinct patterns per domain)
```

关键发现：
- **中间层和高层的领域分化最明显**：低层路由权重跨领域均匀分布，中高层专家权重呈现清晰的领域偏好模式
- **arXiv 和 Python code 的路由权重更相似**：可能因为 LaTeX 代码和 Python 代码都与自然语言有距离
- **与 token-level 专家行为完全不同**：token-level MoE 学到的模式是"专家 3 处理冠词、专家 5 处理标点"（浅层），Lory 学到的是"专家 7 处理 arXiv 论文、专家 2 处理 Python 代码"（深层语义）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
- **无需领域标签**：专家在没有领域监督的条件下自动学习领域专业化，完全通过 self-supervised language modeling + similarity-based data batching 驱动。
- **Similarity-based batching 是必要条件**：random batching 下段级路由学到的专业化程度不足（图 4），因为相邻段来自不相关的文档，路由信号被稀释。
- **互补性应用前景**：论文建议将 segment-level 的领域特征与 token-level 的语法特征结合，构建更强模型。例如，在同一 MoE 层中同时使用两种粒度的路由。
- **Out-of-domain 泛化**：在训练数据中低频的领域（如 Python code 在 CommonCrawl 中占比小），segment-level routing 通过领域级特化专家提供更强的 out-of-domain 性能（Python perplexity 12.5 vs EC 的 14.1/13.6）。
- **Expert Utilization**：无辅助负载均衡损失仍能实现高专家利用率（图 9），专家之间的自然领域专业化自发实现负载均衡（不同领域 token 自然路由到不同专家组）。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training
