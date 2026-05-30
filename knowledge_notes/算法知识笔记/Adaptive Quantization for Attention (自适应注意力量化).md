## Adaptive Quantization for Attention (自适应注意力量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Quantization for Attention 是 SageAttention 的 per-layer kernel 选择策略。不同 layer 对 INT8 PV 量化敏感度不同。离线校准：(1) 对每层测试 SAGEAttn-vB 的 cosine similarity；(2) >99.8%（SAGEAttn-B 最差 cosine sim）→ 用 vB（全 INT8, +4% speed）；(3) 否则回退 SAGEAttn-B（FP16 PV）。adaptive 策略比纯 SAGEAttn-T 提升 11.7% OPS，零指标损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线校准
kernel_map = {}
for layer_id in range(num_layers):
    cos_sim = measure(SAGEAttn-vB, calibration_data[layer_id])
    kernel_map[layer_id] = 'vB' if cos_sim > 0.998 else 'B'
# 在线推理: O = sage_attention(Q,K,V, kernel=kernel_map[layer_id])
```
与 LLM.int8() 混合精度不同：adaptive 在 kernel 粒度做选择（整层统一），避免 kernel 内条件分支。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
离线校准一次完成，推理时仅常数时间 kernel dispatch。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
