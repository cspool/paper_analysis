## Post-Training Quantization (PTQ / 训练后量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Post-Training Quantization (PTQ) 是不需要重新训练即可将全精度模型转换为低精度模型的技术。对于 LLM，PTQ 分为两大类别：(i) Weight-Only Quantization——仅量化权重（W4A16、W3A16），激活保持 FP16，推理时动态反量化到 FP16 进行 MatMul；(ii) Weight-Activation Co-Quantization——同时量化权重和激活（W8A8、W4A8），可直接利用整数计算单元加速。

标准量化公式：

$$X^{\text{Int}N} = \text{Round}\left(\frac{2^N}{\text{absmax}(X^{\text{FP32}})} \times X^{\text{FP32}}\right) = \text{Round}(c^{\text{FP32}} \times X^{\text{FP32}})$$

$$X^{\text{FP32}} = \text{dequantize}(c^{\text{FP32}}, X^{\text{Int}N}) = \frac{X^{\text{Int}N}}{c^{\text{FP32}}}$$

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GPTQ（Weight-Only PTQ 核心方法）的逐层量化流程：

```
# GPTQ: Layer-wise quantization with Hessian-based error compensation
for layer in model.layers:
    W = layer.weight                               # [d_out, d_in] FP16
    H = inverse_hessian(W, calibration_data)        # [d_in, d_in]
    # H captures weight importance correlations
    
    for col in range(d_in):
        # Quantize column 'col'
        w_q[:,col] = quantize(W[:,col])  # INT3/INT4
        error = w_q[:,col] - W[:,col]
        # Compensate: update remaining columns using Hessian
        W[:,col+1:] -= error * H[col, col+1:] / H[col, col]
    # Result: W_int4 with compensation minimizing output error
```

AWQ（Activation-aware Weight Quantization）的关键改进：观察 activation 分布而非 weight 大小来决定哪些权重重要。通过 per-channel scaling factor $s$ 保护~1% salient weights：

$$s^* = \arg\min_s \|\| Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X - WX \|\|$$

SmoothQuant（Weight-Activation Co-Quantization）：利用 activation 不同 channel 的相似性，通过 per-channel scaling 变换将量化难度从 activation 转移到 weight：$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

主流实现：GPTQ（GitHub: IST-DASLab/gptq）、AWQ（GitHub: mit-han-lab/llm-awq）、llama.cpp（K-quant 量化 2-8bit，CPU 3-4× 加速）、bitsandbytes（LLM.int8() 8-bit 量化 GPU 推理）、TensorRT-LLM（集成 GPTQ/AWQ/SmoothQuant）。QuaRot 和 SpinQuant 利用随机旋转矩阵消除 outlier 提高量化友好度。VPTQ 引入 Vector Quantization 替代标量量化，在 2-bit 下相比 GPTQ/AWQ 提升 up to 4.41 perplexity。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
