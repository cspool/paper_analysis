## Online Importance Learning (在线重要性学习)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Online Importance Learning 是 AdaSkip 在 Decoding 阶段用于发现额外 FFN sublayer 跳过机会的方法。基于 Observation 3（FFN 在 decoding 阶段 IO Similarity 高于 prefill），利用当前上下文前 P 个 decoded token（online learning window）的 IO Similarity 信息，动态识别出高于离线学习阈值 β 的额外 FFN sublayer，加入 skip set 以进一步加速 decoding。

核心 Insight（Table 2）：仅用初始少量 decoded token 就能高命中率预测后续 decoding 中不重要 sublayer。Window size 从 5→20 时 hit rate 显著提升，20→40 趋稳，P≈20 即可。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Input: offline 学习的 skipped_set, P=online learning window size
// Step 1: 前 P 个 token 全 sublayer 执行
for token t in 1..P:
    for sublayer j in all sublayers:
        output = forward_sublayer(j, input)
        if sublayer_type_j == FFN:
            Simi_j^P += cosine_sim(input_j, output_j)

// Step 2: 平均
for FFN j: Simi_j^P /= P  // 公式(5)

// Step 3: 阈值 β
β = min(Simi_j for j in skipped_set)

// Step 4: 找出当前上下文高于 β 的额外 FFN
for FFN j not in skipped_set:
    if Simi_j^P > β:
        extra_skipped.append(j)

// Step 5: 合并 skip set
skipped^P = skipped_set ∪ extra_skipped
// 第 P+1 token 起用 skipped^P 跳过
```

术语一般如何实现？如何使用？

实现要点：P≈20 即可获得稳定 hit rate；仅对 FFN 做 online learning（attention skipping 完全由 offline 确定）；online overhead 仅 P 个 token 的全量 forward。额外 FFN skip 同样使用 offline 学习的 Scale_j 补偿。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
