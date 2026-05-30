## Co-referential Token Retrieval in KV-Cache Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Co-referential Token Retrieval 指在长文本解码过程中，模型需要回溯并精确检索之前在上下文中出现过的特定实体（如人名、地名、术语）的能力。在 KV-cache 稀疏化场景中，这是一个核心挑战：较早出现的 co-referential token 可能在当前步的注意力分数较低（因为它与中间插入的 distractor 内容无关），但在未来的查询步中可能突然变得非常重要。例如：对话开头提到 "wraithspire" 这个地点名，中间插入大量无关内容，当后续问题问及地点时，"wraithspire" 的每个 token 都需要被完整检索到。TokenButler 论文设计了一个 synthetic co-reference benchmark（100 个虚构地点名，10^8 组合空间）专门测试这一能力，评估 accuracy（所有 token 被完整检索的比例）和 coverage（被检索到的 token 比例）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Co-referential retrieval 的失败模式分析：
- **驱逐型方法（H2O, SnapKV）**：token 在当前步注意力分数低时被永久驱逐，后续无法检索 → accuracy 仅 1-10%（Llama-8B）
- **分页方法（Quest）**：co-referential 实体的 token 可能跨越 page 边界，page 级别选择可能丢失部分 token → coverage 仅 19-58%
- **Oracle（理想）**：49-81% accuracy（上限取决于模型本身能力）
- **TokenButler**：逐 token 细粒度选择 + 不驱逐任何 token → accuracy 48-80%，接近 Oracle

```python
# Co-referential benchmark 样本结构
sample = {
    "contextual_lead": "Shrouded in fog, place is:",
    "location": "wraithspire",          # 需要被检索的实体
    "philosophical": "...distractor...",
    "culinary": "...distractor...",
    "math_problem": "...distractor...",
    "location_prelude": "Which location up the shore?",
    "answer": "wraithspire"            # 期望输出
}
# 评估：检查 "wraithspire" 的所有 token 是否完整被 attention 机制访问到
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 benchmark 使用 OpenAI gpt-4o-mini 生成 100 个独立组件（地点名、上下文引导语、哲学陈述、烹饪描述、数学问题），随机组合生成测试样本。评估时不依赖生成质量（不检查输出正确性），而是直接检查 attention mask 中的 token selection accuracy 和 coverage — 这是 token 级别的检索精度衡量。可用于评估任何 KV-cache 稀疏化方法在 retrieval-intensive 场景下的表现。

涉及论文标题：
- TokenButler: Token Importance is Predictable
