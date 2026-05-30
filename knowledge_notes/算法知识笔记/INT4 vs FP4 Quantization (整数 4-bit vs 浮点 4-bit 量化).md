## INT4 vs FP4 Quantization (整数 4-bit vs 浮点 4-bit 量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT4 和 FP4 是两种 4-bit 量化数据格式。INT4 均匀映射 16 个等间距整数（[-8, 7]），每个值之间步长相等。FP4（E2M1 格式）由 1-bit 符号 + 2-bit 指数 + 1-bit 尾数组成，仅 15 个有效值（因正负零冗余），可表示 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}。两者在 W4A4 QAT 场景中精度接近，但 INT4 在 per-channel/token 粒度下优 0.015 loss（因多 1 个可表示值），在 group-wise 下等价。论文选择 INT4 作为缩放定律实验的默认格式（因等价或更优且数学形式更简单），并假设 INT4 和 FP4 遵循相同的缩放定律函数形式（图 13 实验验证 INT4 拟合的缩放定律能准确预测 FP4 误差趋势）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# INT4 量化/反量化
scale = 8 / max(|X|)  # M=8 for INT4
X_int = clamp(round(X/scale), -8, 7)
X_hat = X_int * scale  # 均匀分布值: {-8s, -7s, ..., 7s}

# FP4 E2M1 量化/反量化
# E2M1: S(1bit) | E(2bit) | M(1bit)
# bias = 2^(2-1) - 1 = 1  (但E2M1通常bias=0 for subnormal)
# Normal values: (-1)^S × 2^{E} × (1+M/2) for E>0
# Subnormal: (-1)^S × 2^{0} × (M/2) for E=0
# 15 unique values (0 有 ±0 冗余)
scale = 6 / max(|X|)  # M=6 for E2M1 FP4
X_fp4 = map_to_nearest_fp4(X/scale)
X_hat = X_fp4 * scale
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
INT4 广泛用于 LLM 量化（GPTQ、AWQ、GGUF Q4_0 等），因硬件支持成熟（NVIDIA INT8 Tensor Core 可模拟 INT4×INT4 GEMM）。FP4 随着 Blackwell GPU 原生 FP4 Tensor Core 的推出受到更多关注。选择建议：(1) INT4 在细粒度下略优（1 个额外可表示值）；(2) FP4 的 E2M1 隐式匹配分布形状——对小值区域分配更密集层级（0, 0.5, 1, 1.5, 2），类似 Student t-distribution 的尖峰厚尾特征；(3) FP4 可通过 supernormal support 回收负零位提升精度。

涉及论文标题：
- Scaling Law for Quantization-Aware Training
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
