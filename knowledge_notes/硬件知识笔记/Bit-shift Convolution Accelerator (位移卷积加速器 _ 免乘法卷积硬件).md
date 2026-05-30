## Bit-shift Convolution Accelerator (位移卷积加速器 / 免乘法卷积硬件)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-shift Convolution Accelerator 是一种利用权重量化为 2 的幂次值后，将卷积中的乘法替换为 bit-shift 操作的自定义硬件加速器设计。当 CNN 权重被量化为 shift quantization 或 FQ 格式（即 `v = s * 2^{e-b}`），卷积计算 `a * v = a * s * 2^{e-b}` 等价于：
- `s > 0`: 将激活值 `a` 左移 `(e-b)` 位（当 e≥b）或右移 `(b-e)` 位（当 b>e）
- `s < 0`: 同上但取负数（补码运算）
- `s = 0`: 跳过（pruned weight）

这使得卷积的 MAC（Multiply-Accumulate）操作退化为 SAC（Shift-Accumulate），完全消除乘法器阵列——乘法器是数字电路中面积和功耗最大的组件之一。FQ 论文的加速器设计评估了一个 3×3 卷积层（padding=1, 8×8×100 输入/输出），在 unrolled architecture + 相同吞吐量假设下，5-bit FQ 加速器仅需 275.6M 双输入逻辑门，与 3-bit 标准 shift quant (275.2M) 持平，远少于 ABC-Net (806.1M, 2.93×) 和 LQ-Net (314.4M, 1.14×)。

FQ 硬件高效的关键设计：
1. μ_c 量化为 2 的幂次值，保持整体 shift 性质
2. σ_+ = σ_- 约束相等，可融入逐层 α
3. α 进一步融入 BN 融合，消除推理时的最终乘法
4. 无需 N 路并行二值卷积（ABC-Net/LQ-Net 需要 O(MN) 高精度乘积累加）

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FQ Bit-shift Convolution Accelerator 的 dot-product 数据路径（Figure 4）：

```
Input: a[0..R*C-1]  (activation pixels, integer)
       w[0..K-1]     (filter weights, FQ format)
       K = Cin * Kh * Kw

For each output position (single dot-product):

┌─────────────────────────────────────────────┐
│  Activation    │   FQ Weight    │  Compute   │
│  a_i (int)     │  s * 2^{e-b}  │            │
├────────────────┼────────────────┼────────────┤
│  Step 1: Weight Decode                      │
│    sign = w[i].sign   (1 bit)               │
│    exp  = w[i].exp    (3 bits)              │
│    comp = w[i].comp   (1 bit, ± cluster)     │
│    μ_c, σ_c ← from component comp           │
│                                              │
│  Step 2: Effective Scale                    │
│    shift_amt = exp - b                       │
│                                              │
│  Step 3: Bit-shift & Accumulate             │
│    if sign == 0: skip (pruned)              │
│    elif shift_amt >= 0:                     │
│      partial = a_i << shift_amt             │
│    else:                                     │
│      partial = a_i >> (-shift_amt)          │
│    if sign < 0: partial = -partial          │
│    acc += partial                            │
│                                              │
│  Step 4: Final Scale (per output channel)   │
│    output = acc * α  (fused into BN)         │
└─────────────────────────────────────────────┘
```

**Annotations**: `shift_amt` 由 weight 的 exponent 和 layer bias 决定；logical right shift 用于正数，arithmetic right shift 用于负数（保留符号）；FQ 的 σ_c 和 μ_c 已在权重解码时应用（`θ_hat = α * (s*2^{e-b} * σ_c + μ_c)`），σ_+ = σ_- 使 component 间共享 scale；最终 α 乘以累加和可融入 BatchNorm 的 γ/β 参数中，消除额外乘法器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit-shift convolution accelerator 的实现方式：

1. **FPGA 实现**：Mayo 框架（https://github.com/deep-fry/mayo, ICFPT 2019）提供自动多精度多算术 CNN 加速器生成工具。从量化模型描述生成 FPGA 上的展开或折叠式数据路径。FPGA 的 LUT 可高效实现移位逻辑和整数加法。

2. **ASIC 实现**：自定义数据路径设计，使用 barrel shifter + integer adder 替代乘法器。Barrel shifter 的面积远小于同精度乘法器。FQ 论文的 gate-count 估算基于 ASIC 双输入逻辑门下界。

3. **CPU/GPU 实现**：利用 SIMD 整数移位指令（如 x86 `PSLLW`/`PSRAW`, ARM `SSHL`/`USHL`），在标准处理器上也能获得加速，但不如专用硬件高效。

4. **FQ 特定优化**：σ_c 和 μ_c 的 scale factor 在硬件中可通过预计算查找表或简单移位实现（因为 μ_c 也是 2 的幂次）。最终 scaling 融合到 BN 中，消除推理时所有乘法。

Bit-shift 加速器最适合部署在功耗和面积受限的边缘/IoT 设备上（FQ 论文的目标场景），因为消除乘法器阵列可显著降低功耗和芯片面积，同时 bit-shift 操作的延迟远低于乘法。

涉及论文标题：
- Focused Quantization for Sparse CNNs

---
