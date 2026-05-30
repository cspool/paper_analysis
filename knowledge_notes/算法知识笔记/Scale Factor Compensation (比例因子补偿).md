## Scale Factor Compensation (比例因子补偿)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Scale Factor Compensation 是 AdaSkip 中用于补偿 sublayer 跳过造成信息损失的技术。当 sublayer 被跳过时，输出直接等于输入（identity shortcut），但会丢失 sublayer 对向量模长的变换。Scale Factor 补偿通过将输入按历史平均模长比缩放，使近似输出更接近原始输出。

$$Scale_{j} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{|T_{i}|} \frac{\|\vec{b}_{it}^{j}\|}{\|\vec{a}_{it}^{j}\|}}{\sum_{i=1}^{N} |T_{i}|}$$

跳过时近似输出：$\vec{b}_{it}^{\hat{j}} = Scale_j \cdot \vec{a}_{it}^j$

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 在 Offline Importance Learning 中同时累积
Scale_j = average(||output|| / ||input||) across all historical samples

// 推理时 skip sublayer 应用补偿
if j in skipped_set:
    b_hat = Scale_j * a    // 跳过 + 模长补偿
else:
    b = forward_sublayer(j, a)
```

有效性前提：Residual connection 使输入输出模长变化微小，当夹角不大时（高 Similarity），仅模长缩放即可有效补偿。高 Similarity = 小夹角 = Scale Factor 补偿有效。

术语一般如何实现？如何使用？

Scale Factor 与 IO Similarity 一起在 Offline Importance Learning 阶段被计算和存储。每个 sublayer 对应一个 Scale_j。使用时仅需 bypassed sublayer 的输入乘以 Scale_j 作为近似输出——O(1) 乘法 per skipped sublayer per token，额外开销可忽略。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference
