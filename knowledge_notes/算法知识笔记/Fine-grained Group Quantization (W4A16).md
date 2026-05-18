## Fine-grained Group Quantization (W4A16)

术语是什么？通过联网搜索让回答具体和精准。

Fine-grained Group Quantization是一种LLM权重量化方法，将权重矩阵按较小的group（如group size=32或128）分组，每组独立计算scale和zero-point进行量化。与per-tensor/per-channel量化不同，group quantization的量化参数更精细（group内统计更均匀），因此能在4-bit等低比特下保持更高模型精度。典型方案如GPTQ、AWQ、llama.cpp的Q4_0均采用group quantization。W4A16表示weight 4-bit、activation 16-bit（FP16）的混合精度配置，activation保持浮点避免精度损失，仅在compute前做runtime weight dequantization。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以Q4_0 symmetric quantization (group size=32)为例：
```
// Quantize: weight matrix W (FP16) → quantized W_q (INT4) + scales (FP16)
for group g in range(num_groups):  // 沿column dim, 每组32个元素
    max_abs = max(abs(W[g*32 : (g+1)*32]))
    scale = max_abs / 7.0            // Q4_0: [-7,7] range, symmetric
    W_q[g*32 : (g+1)*32] = round(W[g*32 : (g+1)*32] / scale)
    scales[g] = scale

// Dequantize at runtime:
for group g:
    W_fp16[g*32 : (g+1)*32] = W_q[g*32 : (g+1)*32] * scales[g]
// Then perform FP16 GEMM: O = A × W_fp16
```

在移动NPU上，QNN仅支持per-tensor/per-channel粗粒度量化：Llama3.2-1B-Instruct W4A16在MATH500上的accuracy从AutoAWQ group quantization的15.9骤降至2.1，证明fine-grained quantization对数学推理任务至关重要。论文通过hardware-aware tile quantization将group quantization的layout重排为HMX tile format，既保留精度又适配硬件。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：llama.cpp的GGUF格式原生支持Q4_0、Q4_1、Q5_0、Q8_0等多种group quantization格式，通过llama-quantize工具转换。AWQ (AutoAWQ)通过calibration data计算per-channel scaling factor并以group方式存储。GPTQ使用二阶信息逐列量化。在runtime，dequantization通常由GEMM kernel内的vector/SIMD指令完成：先加载scale→broadcast→乘quantized values→accumulate。开销取决于group size（越小越精细但dequant overhead越大）。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

