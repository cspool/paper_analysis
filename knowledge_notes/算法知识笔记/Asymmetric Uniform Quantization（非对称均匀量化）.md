## Asymmetric Uniform Quantization（非对称均匀量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric Uniform Quantization 是一种将浮点张量映射到低比特整数空间的量化方案。与对称量化（symmetric quantization，zero-point=0）不同，非对称量化引入一个可调零值偏移（zero-point z），使量化区间不关于 0 对称。数学定义：

$$\hat{x} = \text{clamp}\left(\left\lfloor \frac{x}{s} \right\rceil + z,\ 0,\ 2^b - 1\right)$$

其中：
- s（scale）= $(x_{\text{max}} - x_{\text{min}}) / (2^b - 1)$，量化步长
- z（zero-point）= $\lfloor -x_{\text{min}} / s \rceil$，将浮点 0 映射到的整数值
- b = 量化位宽（如 4 for INT4）
- ⌊·⌉ = 四舍五入

反量化：$\tilde{x} = s \cdot (\hat{x} - z)$

非对称量化的优势：当数据分布不以 0 为中心时（如 ReLU 激活输出、KV cache 中的 V tensors），可以更充分利用量化区间，减少 clipping error。代价：需要额外存储 zero-point（与 scale 量级相当），且 dequantization 操作多一个减法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XStreamVGGT 中非对称均匀量化的具体实现：

```
def asymmetric_quantize(x, b=4):
    """
    x: 浮点张量，shape 任意
    b: 量化位宽，默认 4 (INT4)
    返回: x̂ (量化值), s (scale), z (zero-point)
    """
    x_min, x_max = x.min(), x.max()

    # Scale: 量化区间均匀划分
    s = (x_max - x_min) / (2**b - 1)

    # Zero-point: 将浮点0映射到的整数位置
    z = torch.round(-x_min / s)
    z = torch.clamp(z, 0, 2**b - 1)   # zero-point也需在有效范围内

    # 量化: round(x/s) + z, clamp到[0, 2^b-1]
    x̂ = torch.round(x / s) + z
    x̂ = torch.clamp(x̂, 0, 2**b - 1)

    return x̂.to(torch.uint8), s, z

# 反量化
def asymmetric_dequantize(x̂, s, z):
    return s * (x̂.float() - z)
```

数值示例（INT4, b=4, 范围 0-15）：
- x = [-2.0, 0.0, 3.0, 5.5] → x_min=-2.0, x_max=5.5
- s = (5.5-(-2.0))/15 = 0.5
- z = round(2.0/0.5) = 4
- x̂ = [0, 4, 10, 15]
- 反量化: x̃ = [-2.0, 0.0, 3.0, 5.5] ✓ (无量化误差，此例恰好完美恢复)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过 `torch.quantize_per_channel` / `torch.quantize_per_tensor` 配合 `torch.qint8` / `torch.quint8` 等 dtype 实现。对于自定义位宽（如 INT4），使用手动 round + clamp。KIVI 库提供完整的 INT2/INT4 asymmetric quantization pipeline（group-wise or channel-wise）。在 GPU kernel 层面，量化和反量化通常在 attention kernel 外完成（如 XStreamVGGT 的方案），但也有 fused kernel 方案将 dequantization 嵌入 attention 计算以减少 memory access（如 BitDecoding）。XStreamVGGT 中使用 group size=64，即每 64 个元素共享一组 (s, z) 参数。

涉及论文标题：
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression
