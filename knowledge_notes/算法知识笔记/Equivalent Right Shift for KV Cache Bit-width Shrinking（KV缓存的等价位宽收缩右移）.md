## Equivalent Right Shift for KV Cache Bit-width Shrinking（KV缓存的等价位宽收缩右移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Equivalent Right Shift 是 PM-KVQ 为渐进式量化中的位宽收缩步骤设计的移位策略。当 KV Cache 需要从 2b-bit 降级到 b-bit 时（如 8→4 bit），该策略通过整数加法和移位操作实现与"先反量化为浮点→再量化到更低精度"等价的效果，避免浮点计算开销。核心公式：X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b。

从算法pipeline角度拆解术语：

**三种位宽收缩策略对比（以 4-bit→2-bit 为例）**：

```
假设 4-bit 量化值 {0,1,...,15} → 压缩到 2-bit {0,1,2,3}

原始 4-bit 量化参数：
  S_4, Z_4  // 缩放因子和零点
  X_4 = quantize(X_fp16, bits=4, S=S_4, Z=Z_4)

// === (a) Direct Right Shift ===
// 直接右移 2 bits，仅保留高位
X_2 = X_4 >> 2  // {0..3}→0, {4..7}→1, {8..11}→2, {12..15}→3
Z_2 = Z_4
S_2 = (2^2 + 1) * S_4 = 5 * S_4
// 问题：信息丢失严重，pass@1 从 44.17% 降至 12.08%

// === (b) Modified Right Shift ===
// 修改零点和缩放因子以保持均值映射
X_2 = X_4 >> 2
Z_2 = Z_4 - (2^{2b} / 2) * S_4  // 调整零点
S_2 = 2^2 * S_4 = 4 * S_4
// 问题：仍丢失信息，pass@1 = 28.75%

// === (c) Equivalent Right Shift (PM-KVQ) ===
// 等价于 dequantize→requantize，但全整数操作
// X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b
b = 2, 2b = 4:
X_2 = ((2^4 - 2^2 + 1)(X_4 + 2^{1})) >> 12
    = ((16 - 4 + 1)(X_4 + 2)) >> 12
    = (13 * (X_4 + 2)) >> 12

// 对应等价浮点操作：
// 1. 反量化到 FP16:  X_fp16 = X_4 * S_4 + Z_4
// 2. 重新量化到 2-bit: X_2_new = clamp(round((X_fp16 - Z_2) / S_2), 0, 3)
// 3. Equivalent Right Shift 在整数域产生相同的量化值
Z_2 = Z_4  // 零点不变
S_2 = (2^b + 1) * S_4  // 例如 b=2: S_2 = 5*S_4
// 效果：pass@1 = 38.33%（vs Direct的12.08%, Modified的28.75%），Voting lossless
```

**公式推导（2b→b bit, b∈{2,4,8}）**：
- 16-bit→8-bit: b=8, 2b=16 → X_8 = ((2^{16}-2^8+1)(X_{16}+2^7)) >> 24
- 8-bit→4-bit: b=4, 2b=8 → X_4 = ((2^8-2^4+1)(X_8+2^3)) >> 12
- 4-bit→2-bit: b=2, 2b=4 → X_2 = ((2^4-2^2+1)(X_4+2^1)) >> 6

术语一般如何实现？如何使用？

实现关键：(1) 仅需整数乘法和移位，在 GPU/CPU 上均可高效执行（single cycle per element）；(2) 零点不变 (Z_b = Z_{2b}) 简化实现；(3) 缩放因子放大 (S_b = (2^b+1)S_{2b}) 补偿位宽降低后的动态范围损失；(4) 渐进式量化中仅在位宽收缩节点执行，不产生每步开销。

局限性：该策略设计用于 2 的幂位宽。对于非 2 的幂位宽（如 3-bit, 6-bit），需要不同的收缩策略。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
