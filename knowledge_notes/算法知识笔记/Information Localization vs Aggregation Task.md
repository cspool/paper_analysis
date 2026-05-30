## Information Localization vs Aggregation Task

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Information Localization vs Aggregation 是 WindowKV 提出的长上下文下游任务二分法，驱动 KV cache 窗口选择的 p 参数：

1. **Information Localization**（信息定位）：QA 类任务——在长文本中定位关键段落并回答。需完整窗口语义 → p=ω（保留窗口中全部 token）。示例：NarrativeQA, Qasper, HotpotQA, 2WikiMQA, Musique。

2. **Information Aggregation**（信息聚合）：摘要类任务——从多段落提取显著信息并浓缩。仅需窗口内关键 token → p<ω（仅保留 top-p 高分 token）。示例：GovReport, QMSum, MultiNews, Code completion, Few-shot tasks。

分类器：bert-base-cased, 9551 样本（8:1:1），accuracy 92.69%, recall 95.19%, F1 94.75%。消融验证（Figure 4）：分类错误导致策略-任务不匹配时性能显著下降。

术语一般如何实现？如何使用？

分类器训练：batch_size=16, lr=1e-6, dropout=0.5, 10 epochs, 8×A100 40G。推理时分类器输出 task_type 控制 p 参数，进而影响窗口内 Top-p token 选择和窗口得分 s_k。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
