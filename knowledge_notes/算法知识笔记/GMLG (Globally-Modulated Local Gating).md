## GMLG (Globally-Modulated Local Gating)

术语是什么？
GMLG是MoDES提出的expert重要性评估机制。将离线校准的全局逐层重要性α^{(l)}与推理时的局部routing概率π_i^{(l)}相乘：s_i^{(l)} = α^{(l)} · π_i^{(l)}。α^{(l)}通过KL divergence量化跳过第l层所有expert对final output的影响——浅层α^{(l)}大（对最终输出影响大，应少跳过），深层α^{(l)}小（可多跳过）。Inference时α^{(l)}已预计算，s_i^{(l)}仅需一次乘法——零额外开销。

从算法pipeline角度拆解术语：
```
// 离线校准 (one-time):
C = 1024 randomly sampled examples from GQA
for each MoE layer l:
    for each example c_j in C:
        prob_j = full_model(c_j)
        prob_j^{(l)} = model_with_layer_l_skipped(c_j)
    α^{(l)} = (1/N) · Σ_j D_KL(prob_j || prob_j^{(l)})
α̃^{(l)} = α^{(l)} / Σ α^{(l')}                     // normalize across layers

// 在线推理 (per token, zero overhead):
s_i^{(l)} = α̃^{(l)} · π_i^{(l)}
// α̃^{(l)}大 → 浅层 → 整体s_i偏高 → 跳过少
// α̃^{(l)}小 → 深层 → 整体s_i偏低 → 跳过多
```

术语一般如何实现？如何使用？
校准在8×H200上执行，每层需2次forward pass。校准数据鲁棒——GQA/COCO/VMMMU上α^{(l)}趋势一致，性能差异<1%。α^{(l)}在浅层大、深层小的趋势在所有模型和数据集上一致，且与论文motivation中降低浅层vs深层k值的实验结论一致（浅层降低k性能下降更严重）。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---
