## Shift Quantization (幂次量化 / Power-of-Two Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shift Quantization（也称 Power-of-Two Quantization）是一种将神经网络权重约束为 2 的幂次值或零的量化方案。一个 (k+2)-bit 的 shift quantization 表示值为 `v = s * 2^{e-b}`，其中 `s ∈ {-1, 0, 1}` 表示符号（或零），`e ∈ [0, 2^k-1]` 为 k-bit 无符号指数，`b` 为逐层 bias 常数，用于缩放量化值的大小范围。例如 5-bit shift quantization：1-bit 符号 + 3-bit 指数 (k=3) + 1-bit 指示 pruning mask。

核心优势：量化后的权重值均为 2 的幂次，因此卷积中的乘法 `a * v = a * s * 2^{e-b}` 可替换为 bit-shift 操作 `s * (a << (e-b))`（或 `>>` 当 e < b），在硬件中消除乘法器阵列，大幅降低逻辑门数和功耗。

主要局限：量化层级在零附近最密集（±1, ±2, ±4, ±8, ...），距离零越远层级越稀疏。当 CNN 经过细粒度剪枝后，权重分布往往呈现"中空"现象——大量非零权重集中远离零的区间，近零的量化层级利用率极低，造成精度浪费。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Shift Quantization 在 CNN 推理 pipeline 中的执行流程：

```
# Input: weight tensor W ∈ R^{Cout, Cin, Kh, Kw}
#        bias b (layer-wise constant, power-of-two)
#        bit-width n (= k + 2)

# Step 1: Clamp weights to representable range
max_val = 2^{(2^k - 1) - b}
min_val = 2^{0 - b}
for each w in W:
    w_clamped = clamp(|w|, min_val, max_val)

# Step 2: Quantize to nearest power-of-two
for each w in W:
    # Find nearest representable power-of-two
    # Representable values: {0, ±2^{-b}, ±2^{1-b}, ..., ±2^{(2^k-1)-b}}
    v = round(log2(|w|) + b)  # quantized exponent
    v = clamp(v, 0, 2^k - 1)
    w_hat = sign(w) * 2^{v - b}

# Step 3: Convolution with bit-shift instead of multiply
# For activation a and quantized weight w_hat = s * 2^{e-b}:
output += s * (a << (e - b))  # when e >= b
output += s * (a >> (b - e))  # when e < b
```

**Annotations**: `k` 为指数位数（5-bit shift quant 中 k=3）；`b` 为 layer-wise bias，控制整层的 magnitude scale；sign 通过 MSB 编码；量化层级间距随 |v| 增大呈指数增长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Shift quantization 的实现通常在训练后或训练中应用。INQ（Incremental Network Quantization）[26] 是代表性的 shift quantization 训练方法：将权重分组为已量化和未量化两组，逐步将未量化权重量化到最近的 2 的幂次值并冻结，仅微调未量化组。更通用地在 training-aware quantization 流程中，forward pass 使用 `Q^{shift}_{n,b}(w)` 计算量化权重，backward pass 将量化函数视为恒等映射（Straight-Through Estimator, STE）。

Mayo 框架（https://github.com/deep-fry/mayo）提供了完整的 shift quantization 工具链。推理时，bit-shift 替代乘法在 CPU（LEA/SAL 指令）、GPU（整数移位指令）和 FPGA（移位寄存器）上均可高效实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs
