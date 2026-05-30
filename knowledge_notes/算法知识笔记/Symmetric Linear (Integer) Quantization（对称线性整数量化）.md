## Symmetric Linear (Integer) Quantization（对称线性整数量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Symmetric Linear (Integer) Quantization 是一种最基础的均匀量化方案。将浮点张量 $x$ 映射到 k-bit 整数 $x_{int} \in [-2^{k-1}+1, 2^{k-1}-1]$，公式为 $x_{int} = \text{round}(x/s \cdot (2^{k-1}-1))$，其中 scale factor $s = \max(|x|)$（即取张量元素绝对值的最大值）或 per-group 的局部最大值。反量化公式：$x_{deq} = x_{int} \cdot s / (2^{k-1}-1)$。该方案"对称"的含义是量化格点关于零对称（无 zero-point 偏移），INT4 映射到 $\{-7, -6, ..., +6, +7\}$。计算开销极低——仅需一次 max reduction 和一次 element-wise scale+round，延迟在 GPU 上通常为 memory-bound。但对称量化对非对称分布的数据（如 ReLU 激活后的正偏分布或带有 outlier 的梯度）精度较差，因为零值的对称中心约束导致有效量化范围利用不足。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 中使用对称线性量化对权值差值和梯度进行压缩：
```
# INT4 对称量化（group-wise，per-group scale s = max(|x|)）
def symmetric_quantize_int4(x, group_size):
    # x: FP32/BF16 tensor, shape [N]
    for g in range(0, len(x), group_size):
        group = x[g:g+group_size]
        s = max(abs(group))                  # per-group scale
        x_q = round(clip(group, -s, s) / s * 7)  # map to [-7, +7]
        # 通信时发送: packed 4-bit x_q + FP32 scale s
    return packed_x_q, scales

# 反量化
def symmetric_dequantize_int4(packed_x_q, scales, group_size):
    for g, s in enumerate(scales):
        x_deq = packed_x_q[g] * s / 7       # recover FP32
    return x_deq
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对称线性量化广泛用于通信压缩场景（如 QSGD、SDP4Bit），因为在分布式通信中 scale factor 的计算和传输开销小（per-group 仅额外传输 1 个 FP32/FP16 scale）。在 PyTorch 中可用 `torch.quantize_per_tensor`（per-tensor 对称）或手动 group-wise 实现。与反量化结合使用时常见优化：(a) 将 scale 融合到后续的 dequantize-后操作中（如 gradient reduce）；(b) 对于非常小的 group_size（如 32-128），scale factor 存储/传输开销可能不可忽略，需权衡精度和带宽。SDP4Bit 中对权值差值使用 group_size=2048（开销 ≈ 0.1%），对梯度使用 group_size=128（intra）和 512（inter）以更好处理梯度中的局部变动。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
- QSDP: Quantized Distributed Training of Large Models with Convergence Guarantees

---
