## Asymmetric Logarithmic Quantizer (非对称对数量化器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric Logarithmic Quantizer 是 LOGART 提出的首个对数域非对称量化器。传统对数 PTQ 取 |W| 后对称量化，无法匹配 LLM 中非对称权重分布（正负 range 不均）。线性 PTQ 可用 zero-point 偏移解决，但对数域因零附近非线性间距不可行。LogART 通过自适应下界 l_a 为正负值分配不同数量码字：w_h=max(w_max,-w_min), w_l=min(w_max,-w_min), d_a 度量正负 range 在对数域的差异，l_a=⌊d_a/2⌋ 作为量化下界 clamp 起点（从 [0,U] 变为 [l_a,U]）。

从算法pipeline角度拆解术语：
```
# ABS (Asymmetric Bound Search) - calibration-free
for each channel:
    w_h = max(w_max, -w_min); w_l = min(w_max, -w_min)
    if w_l >= t:  d_a = floor(log_sqrt2(w_h)) - round(log_sqrt2(w_l))
    else:  d_a = n1 + floor((m-n1)/2) - round(log2(w_l))
    l_a = floor(d_a / 2)
# 量化 clamp 下界从 0 变为 l_a
```

术语一般如何实现？如何使用？
ABS 为 OHS 的 calibration-free 组件，直接基于 weight 统计量计算，无需任何校准数据。每 channel 独立计算 l_a，额外开销可忽略。消融：OPT-125M PPL 36.10→34.29，LLaMA2-7B PPL 6.56→6.45。对非对称分布明显的 LLaMA 增益更大。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
