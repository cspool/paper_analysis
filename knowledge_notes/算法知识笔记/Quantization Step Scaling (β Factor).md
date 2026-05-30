## Quantization Step Scaling (β Factor)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Step Scaling (β factor / 量化步长缩放因子) 是在计算 uniform quantization 步长时引入的乘法缩放因子 β ≤ 1：δ = β · (max(w) − min(w))/(2^b − 1)。标准 uniform quantizer 使用 β=1，步长由权重最小/最大值决定。但在极低位宽（2-bit、3-bit）下，标准步长导致量化网格过宽，大量权重落于网格边界，量化误差大。通过 β < 1 收缩步长，量化网格更紧密地聚集在权重分布中心，虽然部分 outlier 被裁剪但大多数权重的量化精度提升，总体 MSE 更低。这一现象的理论基础[21,34]是：最低 MSE 的量化步长并不一定由 min/max 决定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 标准步长（β=1）
δ_std = (max(w) - min(w)) / (2^b - 1)

# β-scaled 步长
δ_magr = β · δ_std

# MagR 经验取值:
#  per-channel INT2: β ∈ [0.80, 0.85]
#  per-channel INT3: β ≈ 0.90
#  per-channel INT4: β = 1.00
#  per-group INT2/INT3: β = 0.95

# 量化过程与标准 uniform quantizer 完全相同，仅 δ 不同
w_q = δ_magr · clamp(round(w/δ_magr) - z, 0, 2^b-1) + z·δ_magr
```

β 与 bit-width 正相关：位宽越低，标准步长的缺陷越明显，需要更大幅缩放。INT4 有 16 个量化级别，标准步长通常已足够好。MagR 预处理降低权重最大幅度后，β 进一步收缩步长将量化网格对齐到高密度区域，两者协同大幅降低 sub-4bit 误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
β 通过网格搜索在 calibration 集上选择：完成 MagR 预处理 → 对 β ∈ {0.80, 0.85, 0.90, 0.95, 1.00} 分别量化 → 选最低 perplexity 对应的 β。MagR ablation study（Table 8）显示 INT2 下 β=0.80 PPL=16.73 vs β=1.00 PPL=16.99，INT3 下 β=0.90 PPL=6.41 vs β=1.00 PPL=6.43。该技术也被 TWN[21]、XNOR-Net[34] 等早期工作中观察到。β 与 MagR 是正交增强：MagR 缩小权重范围，β 进一步优化步长。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization
