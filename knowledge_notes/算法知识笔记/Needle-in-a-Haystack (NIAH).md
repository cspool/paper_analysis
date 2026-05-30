## Needle-in-a-Haystack (NIAH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Needle-in-a-Haystack (NIAH, Kamradt 2023) 测试 LLM 长上下文检索：在大量无关文本中嵌入关键信息（needle），要求模型在所有长度和深度位置检索并复现。Rouge-1 F1 评估。揭示了 "lost-in-the-middle" 现象——中间位置准确率大幅下降。

在 KV cache 压缩评估中广泛使用。WindowKV 在 LLaMA3-8B, context=8K, KV size=512 下评估，热力图显示 WindowKV 在所有深度位置的检索准确率优于 StreamingLLM/H2O/PyramidKV。

术语一般如何实现？如何使用？

Needle 放置在指定深度百分比（0%-100%），格式如 "The pass key is <N>."。Haystack 为重复填充文本。Rouge-1 F1 匹配 needle 原文。WindowKV 中 context=8K, KV size=512。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference
