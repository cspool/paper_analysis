## Layer-wise Skipping (逐层跳过)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-wise Skipping 是一种在 LLM 推理过程中跳过 Transformer Layer 执行的加速策略，通过省略特定位置层的 Self-Attention 和 FFN 计算（两模块同时跳过）来减少计算量和延迟。该方法基于关键观察：并非所有 Transformer 层对生成质量同等重要，某些层的输出与其输入高度相似（高 IO Similarity），可以被安全跳过。

AdaSkip 论文将现有 Layer-wise Skipping 策略分为三类：
1. **Early Skipping (SkipDecode)**：始终跳过模型前几层（除第一层外）。支持 batching 操作但可能跳过重要层。
2. **Periodic Skipping (Unified Skipping)**：在中间层按固定频率周期性跳层（如每 N 层跳 1 层）。支持 batching 但无法捕捉不同层的重要性差异。
3. **Early Exit**：在每层计算后判断条件（如置信度），一旦满足条件立即退出，跳过后续所有层。可能忽略后面更重要的层，且通常需要额外训练 classifier 或微调模型。

Layer-wise Skipping 的核心限制：按整层（attention + FFN）粒度进行 skip，忽略了 attention sublayer 和 FFN sublayer 各自独立的重要性分布特征。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Early Skipping (SkipDecode) 伪代码**：
```
for layer l in 1..L:
    if l <= K and l != 1:
        skip  // identity shortcut
    else:
        x = Attention(LayerNorm(x)) + x
        x = FFN(LayerNorm(x)) + x
```

**Periodic Skipping (Unified Skipping) 伪代码**：
```
for layer l in 1..L:
    if l > 1 and l < L and (l - 1) % N == 0:
        skip  // identity shortcut
    else:
        x = Attention(LayerNorm(x)) + x
        x = FFN(LayerNorm(x)) + x
```

**Early Exit 伪代码**：
```
for layer l in 1..L:
    x = Attention(LayerNorm(x)) + x
    x = FFN(LayerNorm(x)) + x
    if confidence_score(x) > threshold:
        return lm_head(x)
return lm_head(x)
```

术语一般如何实现？如何使用？

Layer-wise Skipping 通常以 hook 或 model patching 方式集成到 HuggingFace Transformers 推理流程中。实现可以是：替换指定层的 forward 方法为 identity function（直接返回输入），或在 forward 入口处判断是否属于 skip set。Early Exit 变体需要额外训练 confidence classifier 或微调模型参数以补偿信息损失。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
