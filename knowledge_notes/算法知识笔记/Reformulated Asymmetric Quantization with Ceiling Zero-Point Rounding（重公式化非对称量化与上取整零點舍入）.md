## Reformulated Asymmetric Quantization with Ceiling Zero-Point Rounding（重公式化非对称量化与上取整零點舍入）

术语是什么？

GyRot提出的非对称量化公式改造：将传统先scale后bias的公式 x̂=⌊x/s_x+z_x⌉ 改为先bias后scale的 x̂=⌊(x+z_x)/s_x⌉，同时zero-point从scaled domain计算（z_x=−min(x_g)/s_x）改为unscaled domain计算（z_x=−min(x_g)），并使用ceiling ⌈·⌉替代round做ZP量化以避免underflow clipping。

从算法pipeline角度拆解术语：

传统非对称量化在高asymmetry（如HAP后）下的问题：
```
// 传统公式: x̂ = clip(round(x/s_x + z_x), qmin, qmax)
// z_x = -min(x_g)/s_x  (in scaled domain)
// 问题: s_x小 → z_x被放大 → long-tailed ZP distribution (Fig. 5)
// 原因: min(x_g)在HAP后可以很大（biased group distribution）
//       s_x = (max(x_g) - min(x_g))/(2^b-1)
//       当min(x_g)和max(x_g)都很大时, s_x可能很大,
//       但z_x = -min(x_g)/s_x ≈ -(2^b-1) * min/(max-min)
//       若asymmetry严重, min/(max-min) ratio → large → z_x尾部拉长
```

GyRot重公式化：
```
// GyRot公式:
// z_x = ceil(-min(x_g))  // 直接从unscaled domain计算, 无除法放大
// s_x = (max(x_g) + z_x) / (2^b - 1)  // 注意: +z_x而非-min (因为z_x已含符号)
// x̂ = clip((x + z_x) / s_x, qmin, qmax)

// Dequantization (内积):
// y ≈ Σ_g SW[g] * (SX[g] * Σ_i x̂_i·ŵ_i - ZX[g] * Σ_i ŵ_i)
//               ↑ 先乘SX         ↑ 再减ZX×WSUM  ↑ 再乘SW
// vs 传统: y ≈ Σ_g SX[g] * SW[g] * Σ_i (x̂_i - ZX[g])·ŵ_i

// Ceiling ZP rounding:
// ZX_quantized = ceil(zx)  // 保证 ZX_Q ≥ zx, 消除 underflow
// vs 传统round: ZX_quantized = round(zx) → 可能 ZX_Q < zx → clipping (Fig. 6)
```

效果（Table V, LLaMA-3-8B G=32 R=1024, CoRFiG+HAP）：
- 传统asym + Round ZP: FP16 ZP=6.81, INT8 ZP=7.93 (degradation)
- Reformulated asym + Round ZP: FP16 ZP=6.80, INT8 ZP=7.65 (改善)
- Reformulated asym + Ceiling ZP: FP16 ZP=6.81, INT8 ZP=6.91 (near parity)

术语一般如何实现？如何使用？

实现要点：(1) z_x用⌈−min(x_g)⌉而非⌊−min(x_g)/s_x⌉——避免除法放大效应。这对HAP后per-group范围biased的情况尤其重要。(2) s_x公式从(max−min)/(2^b−1)变为(max+z_x)/(2^b−1)，保证量化范围覆盖全部正值。(3) Ceiling rounding在硬件上等同于负数方向的floor：⌈x⌉ = -⌊−x⌋，可用标准整数rounding单元实现。(4) 与fully integer dequantization配合：重公式化使ZX范围可控 → INT8 ZX精度充足 → 全整数dequantization datapath可行。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

