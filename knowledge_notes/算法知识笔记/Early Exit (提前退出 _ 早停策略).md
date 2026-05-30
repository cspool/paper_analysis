## Early Exit (提前退出 / 早停策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Early Exit 是一种 Layer-wise Skipping 策略，在每层 Transformer 计算后评估条件（如置信度 score、entropy、或 classifier 输出），一旦条件满足立即退出推理，跳过后续所有层，用当前 hidden states 直接生成输出。代表工作包括 CALM（Schuster et al. 2022）、LITE（Varshney et al. 2023）、EE-LLM（Chen et al. 2024）。

与 AdaSkip 的 Offline/Online Importance Learning 不同，Early Exit 是条件式动态跳过——跳过的层数不固定。缺点是：(1) 可能跳过后面更重要的层；(2) 通常需要额外训练 classifier 或微调模型；(3) 在长上下文 decoding 中错误逐层累积导致质量显著退化（AdaSkip 实验中 Vicuna Rouge-L 在两个 summarization 任务上降至 <4.0）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Confidence-based Early Exit
for layer l in 1..L:
    x = Attention(LayerNorm(x)) + x
    x = FFN(LayerNorm(x)) + x
    logits_l = lm_head(x)
    confidence = max(softmax(logits_l))
    if confidence > threshold:
        return logits_l
return lm_head(x)

// Classifier-based Early Exit
for layer l in 1..L:
    x = TransformerLayer_l(x)
    exit_prob = classifier_l(x)  // trained per-layer classifier
    if exit_prob > 0.5:
        return lm_head_l(x)
return lm_head(x)
```

术语一般如何实现？如何使用？

实现分类：Confidence-based（无需训练，用 entropy/logit margin 判断，但长文本质量差）；Classifier-based（训练 per-layer classifier，需额外数据）；Fine-tuning-based（如 LITE 的 instruction tuning 使中间层也能生成好输出）。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
