## W4A4KV4 Quantization (4-bit 权重-激活-KV Cache 量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4A4KV4 是 LLM 压缩的激进配置：Weights INT4、Activations INT4、KV Cache INT4。相比 weight-only (W4A16) 仅压缩模型参数，W4A4KV4 同时压缩推理时的激活内存和 KV Cache 内存，大幅减少 decode 阶段显存占用（KV Cache 是长序列生成瓶颈）。在单张消费级 GPU 上运行 7B LLM 成为可能。核心难点：激活和 KV Cache 比权重对量化更敏感，需 Hadamard rotation 等预处理平滑分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
W4A4KV4 单层推理数据流：

```
// Q/K/V 投影: W4A4 GEMM
Q = dequant(W_Q_int4) × dequant(A_Q_int4)  // INT4→FP16→matmul
K = dequant(W_K_int4) × dequant(A_K_int4)
V = dequant(W_V_int4) × dequant(A_V_int4)

// Attention: FP16
S = QK^T/√d,  A = softmax(S)V

// KV Cache: store INT4
K_cache_int4 = round((K - z_K) / s_K)
V_cache_int4 = round((V - z_V) / s_V)

// Output: W4A4
O = dequant(W_O_int4) × dequant(A_O_int4)
```

内存：weight 4×、activation 4×、KV cache 4× 减少；叠加 50% sparsity → weight 再减半（总约 6.4×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuaRot/SpinQuant/FlatQuant 实现 Hadamard rotation + per-channel/token RTN/GPTQ。OBR 叠加 50% sparsity 并加入误差补偿。NVIDIA Ampere/Hopper 原生支持 INT4 Tensor Core。开源：QuaRot (https://github.com/spcl/QuaRot)，OBR (https://github.com/csguoh/OBR)。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
