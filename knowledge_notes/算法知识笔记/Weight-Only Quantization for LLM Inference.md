## Weight-Only Quantization for LLM Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight-Only Quantization 是只量化权重而不量化激活值的后训练量化策略。在 LLM 单 batch 生成式推理中，每 token 解码仅涉及 GEMV 操作（matrix-vector multiply），arithmetic intensity 极低——每个权重加载后只参与一次乘加，无法跨多 token 分摊。推理完全受限于 memory bandwidth 而非 compute。因此只压缩权重（减少内存流量）同时保持 activations 为 FP16，即可获得接近压缩比的加速。SqueezeLLM Sec. 3 用 roofline model 验证：A5000 GPU 上降低 weight 精度→延迟线性下降（Fig. 2），证明 memory 是主导瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线: 量化权重
W_indices_3bit, LUTs, S_csr = quantize(W)  
# 在线推理: activations 保持 FP16
for each Linear layer:
    # Dense: 加载 3-bit indices → LUT查表FP16 → matvec
    Y = lut_dequant_matvec(W_indices, LUTs, X)
    # Sparse (optional): CSR SpMV
    Y += balanced_csr_matvec(S_csr, X) 
# 加速比 ≈ 压缩比 (memory-bound region of roofline)
```

关键前提：(1) 单 batch 推理（batch>1 时 compute 可能成为瓶颈）；(2) GPU memory bandwidth << compute throughput（如 A6000: 768 GB/s vs 222 TFLOPS, ~290x 差距）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表性实现：GPTQ (uniform group quant), AWQ (activation-aware scaling), SqueezeLLM (non-uniform weighted k-means + sparse), SpQR (GPTQ-style + grouping + sparse)。共同特征：activations FP16，计算在 FP16。主要挑战：如何在 3-4 bit 下最小化 perplexity 退化。局限：在 batch_size>1 或 prefill 阶段（compute-bound）加速效果减弱；需要自定义 CUDA kernel 实现 dequantization（PyTorch 原生不支持 3-bit 非均匀格式）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

---
