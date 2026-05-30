## Induction Heads (归纳头)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Induction Heads（归纳头）是 Transformer 模型中一类特殊的功能性 attention head，最早由 Olsson et al. (2022, "In-Context Learning and Induction Heads") 系统描述。Induction heads 的核心行为是 "copy-paste"：当序列中出现模式 [A][B]...[A] 时，induction head 会在最后一个 [A] 处将高 attention weight 分配给紧随前一个 [A] 出现的 token [B]，从而"预测" [B] 应该在此处重复出现。

其底层机制涉及 K-Composition：前一层的 attention head 将 token [B] 的信息写入 token [A] 的残差流中（通过 "previous token head"），当前层的 induction head 通过 Query 匹配 [A] 的 Key，使 [A] 的 Value（包含 [B] 的信息）被高权重读出。这实现了跨位置的模式匹配和复制。

SnapKV 论文在解释为何需要 pooling 聚类时引用了 induction heads（Sec. 4.3）：LLM 的信息检索和生成不仅依赖高 attention weight 的特征本身，还依赖 induction heads 将 attention weight 高的特征周围 token 的上下文信息一并"复制"到输出中。若仅保留孤立的 top attention 位置（不保留周围 token），会破坏 induction heads 的复制机制——例如在电话号码检索中，模型可能仅获取了国家代码却"补全"了错误的其余数字。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Induction Head 的工作机制（简化）
# 模式: 输入序列 "...A B ... A" → 预测 "B"

# Layer L-1 (Previous Token Head):
#   在位置 i 处，将位置 i 的 token 信息写入位置 i+1 的残差流
#   即 token A 的残差流中包含 B 的信息

# Layer L (Induction Head):
#   当前 token 是第二个 "A"
#   Q_A = W_Q @ residual_A  # Query 来自 A
#   K_prefix = W_K @ residual_all  # Key 来自所有位置的残差流
#   scores = Q_A @ K_prefix^T
#   # induction head 给第一个 "A" 的位置高分
#   # → 第一个 "A" 的 Value 中包含 B 的信息
#   # → 输出中复制 B 的信息 → 预测下一个 token 为 B

# SnapKV 的 pooling 设计与 induction heads 的关系:
# 仅 TopK 选择（无 pooling）:
important_positions = TopK(attention_scores, k)  # {100, 200, 350}
# → 位置 99, 101, 199, 201 等相邻 token 的 KV 被丢弃
# → induction heads 无法从 100 的上下文中复制完整信息

# Pooling 聚类（保留邻域）:
pooled_scores = MaxPool1d(attention_scores, kernel_size=5)
important_positions = TopK(pooled_scores, k)  # {99, 100, 101, 199, 200, 201}
# → 相邻 token 被集群保留
# → induction heads 可以正确复制完整上下文
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Induction heads 不是人工设计的模块，而是在 Transformer 预训练过程中自发涌现的功能性电路（circuits）。它们通常在多层 Transformer 中由两层 attention head 组合形成（K-Composition pattern）。在 mechanistically interpretable 的研究中，可通过 activation patching、attention pattern analysis 和 knock-out experiments 来识别。

对于 KV cache 压缩方法设计，induction heads 的存在意味着：(a) 压缩时不能仅保留孤立的高注意力 token，需要保留其邻域 token 以维持复制机制；(b) pooling/clustering 策略（如 SnapKV 的 max pooling 或 PyramidKV 的区块 chunk）是必要的设计选择。实践中 SnapKV 通过 1D max pooling（kernel_size=5~13）实现，PyramidKV 将序列分为固定大小 chunks 并保留完整 chunks。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation

---
