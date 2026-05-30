## Needle In A Haystack (NIAH) / 大海捞针测试

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Needle In A Haystack (NIAH) 是评估 LLM 长上下文检索能力的最广泛使用的 benchmark 之一。测试方法：在长文档（"haystack"——通常由填充文本如 Paul Graham essays 组成）中的特定位置插入一条特定信息（"needle"——如 "The magic number is 12345"），要求模型在给定长文档的情况下正确回答关于 needle 的问题（如 "What is the magic number?"）。评估维度包括：(1) context length（从 1K 到 1M+ tokens）；(2) needle depth（needle 在 context 中的位置，如 0%/25%/50%/75%/100%）；(3) 不同难度的变体（single NIAH, multi NIAH, multi-needle with distractors 等）。

RULER benchmark（Hsieh et al., 2024）将 NIAH 扩展到 13 个 tasks，包含：(a) 原始 NIAH（single needle）；(b) Multi-keys NIAH（多个不同标签的 needle）；(c) Multi-values NIAH（同一标签对应多个 needle）；(d) Multi-queries NIAH（多个问题）；(e) Variable Tracking（追踪变量赋值链）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**NIAH 测试 Pipeline**：
```
# Needle = 关键信息（短文本）
# Haystack = 长填充文本（"The grass is green..." × N times）
# Depth = Needle 插入位置占 context length 的比例 (0-100%)

def evaluate_niah(model, context_length, depth):
    # 1. 构造测试序列
    haystack_prefix = repeat_filler_text(depth * context_length / 100)
    haystack_suffix = repeat_filler_text((100-depth) * context_length / 100)
    prompt = haystack_prefix + NEEDLE + haystack_suffix + QUESTION

    # 2. 推理
    response = model.generate(prompt)

    # 3. 评估
    return contains_needle_info(response, NEEDLE_ANSWER)
    # 返回 0 (失败) 或 1 (成功)
```

术语一般如何实现？如何使用？

NIAH 由于其低 entropy (1.93) 和单点检索特性，是最容易用 sparse attention 完成的任务——Exploiting Sparsity 论文显示 k=1 即可在 1M token context 上达到 100% 成功率。相比之下，Word Counting（需遍历全部文本统计词频）的 attention entropy 为 2.68，需要 8.87% of N 的 k 才能达到 95% dense attention 性能。NIAH 的简单性使其成为验证 long-context retrieval 方法最低可行性的标准，但不应作为长上下文能力的唯一评估依据。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---
