## Fake Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fake Quantization（伪量化/模拟量化）是在训练或校准时模拟低比特精度量化效果的技术。其核心思想是：前向传播时对权重/激活执行完整的量化-反量化操作（quantization-dequantization），使数值被约束在离散的量化级别上；但张量本身仍保持浮点表示，而非真正存储为整数，因此被称为"伪"量化。数学公式为：`v_q = s * round(clip(v, l, u) / s)`，其中 s 为 scale factor。伪量化允许在 FP32 环境中模拟 INT 推理的精度损失，同时保持梯度可以通过 STE 回传。本论文（2DQuant）使用伪量化来模拟 Linear 层和 Batch MatMul 的 INT4/INT3/INT2 算术精度损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 2DQuant 的 DQC 阶段，对 SwinIR 中每个 Linear 层的权重和激活执行伪量化：
```
def fake_quantize(v, l, u, N):
    # N = bit数, 2^N-1 = 量化级别数
    v_c = clamp(v, min=l, max=u)
    scale = (u - l) / (2**N - 1)
    v_r = round((v_c - l) / scale)      # 离散整数值
    v_q = v_r * scale + l                # 反量化回浮点
    return v_q  # 输出仍是 FP32，但值被约束在离散级别

# 前向传播
w_q = fake_quantize(weight, l_w, u_w, bits=4)  # 4-bit 伪量化权重
x_q = fake_quantize(input, l_x, u_x, bits=4)   # 4-bit 伪量化激活
y = linear(w_q, x_q)  # 用伪量化后的值计算，模拟 INT4 精度
```
伪量化的关键效果：原本连续的 FP32 值被强制离散化到 2^N 个候选值上，导致信息损失。DOBI 阶段通过 MSE 搜索找到使 `||v - v_q||_2` 最小的 (l, u)，从而最小化这种信息损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中，伪量化可通过 `torch.quantization.FakeQuantize` 或自定义实现。`torch.fake_quantize_per_tensor_affine` 和 `torch.fake_quantize_per_channel_affine` 提供内置支持。典型用法：`torch.quantization.FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8)`。在 QAT/PTQ 训练中，伪量化模块被插入模型的计算图关键位置（如 Linear 之前和之后），前向时模拟量化，反向时通过 STE 让梯度绕过不可微的 round 操作。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

在 PMQ-VE 中，Fake Quantization 用于模拟多帧视频增强模型（RSTT、MIA、EMA-VFI）的 INT4/INT2 推理精度损失。量化公式：`x_clip = clamp(x, lb, ub)`, `Δ = (ub-lb)/(2^N-1)`, `x_int = round((x_clip-lb)/Δ)`, `x̂ = x_int·Δ + lb`。与 2DQuant 的单一 per-tensor 量化不同，PMQ-VE 为每帧独立使用不同的 (lb_i, ub_i)，量化后的 x̂ 仍保持 FP32 表示但值域被约束在 2^N 个离散级别上。

---
