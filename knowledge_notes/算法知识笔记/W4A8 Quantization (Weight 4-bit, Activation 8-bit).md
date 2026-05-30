## W4A8 Quantization (Weight 4-bit, Activation 8-bit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4A8量化是一种混合精度量化方案：模型权重量化为4-bit整数（UINT4），激活值保持8-bit整数（INT8），推理时执行非对称精度GEMM（Asymmetric GEMM）。相比W8A8，W4A8将权重内存占用减半，降低内存带宽需求，在memory-bound small-batch场景下有优势，并将memory-to-compute转折点batch size减半（H100上从300降至150）；相比W4A16（FP16 activation），W4A8使用INT8 Tensor Core MMA提供更高计算吞吐；相比W4A4，W4A8保持8-bit激活避免激进量化带来的显著精度损失。但W4A8的dequantization（UINT4→INT8）需在GEMM main-loop中通过CUDA Cores完成，若实现不高效会成为瓶颈，使实际性能远低于roofline预测。

从算法pipeline角度拆解术语，给出具体例子。
W4A8 GEMM量化pipeline（两级量化架构）：
```
离线:
  FP16 weight → per-channel quant → INT8 ([-119,119]) → per-group quant → UINT4

在线推理 (main loop per K-tile):
  1. Load UINT4 weight tile from GMEM → SMEM → RF
  2. Unpack 4-bit → 8-bit (register-level)
  3. Dequantize UINT4 → INT8 on CUDA Cores (核心瓶颈)
  4. WGMMA INT8 MMA on Tensor Cores: C += A_int8 × W_int8
  5. Epilogue: INT32 → FP16 (per-channel scale)
```

dequantization开销由三个因素决定：(a) 每元素指令数α，(b) CUDA Cores吞吐Φ_CUDA（远低于Tensor Cores），(c) 权重矩阵大小N×K。在H100上，为与weight loading重叠需α≤5.07，为与MMA重叠需α≤5.05。QServe的QoQ算法（α≥10）无法满足此阈值；LiquidQuant的α≈0.875（含unpack）满足要求。

术语一般如何实现？如何使用？
实现代表：QServe（QoQ algorithm, group=128）、LiquidGEMM（LiquidQuant, group=64）。适用于需平衡精度和内存的LLM serving场景——4-bit权重使大模型可在单GPU运行，8-bit激活保持推理精度。激活量化用SmoothQuant per-token动态量化（FP16→INT8），fuse到前序kernel epilogue消除overhead。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---
