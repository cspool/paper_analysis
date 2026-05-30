## Min-Max Quantization (Uniform Integer Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Min-Max Quantization 是最基础的均匀量化方法，将浮点权重 W 线性映射到 N-bit 整数 {0, 1, ..., 2^N-1}。公式：W̃ = α·⌊(W-β)/α⌉ + β，其中 α = (max(W) - min(W)) / (2^N-1) 为缩放因子，β = min(W) 为零点因子，⌊·⌉ 为整数舍入。这是 QA-LoRA 论文中用于 INT4/INT3/INT2 量化的基本公式。QA-LoRA 将其从 column-wise（α_j, β_j 对每列计算）扩展到 group-wise（α_{l,j}, β_{l,j} 对每组计算），通过增加量化参数数量降低量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Min-Max N-bit 量化
α = (max(W) - min(W)) / (2^N - 1)
β = min(W)
W_hat = round((W - β) / α)        # 量化: FP32 → INT (0 to 2^N-1)
W_tilde = α * W_hat + β           # 反量化: INT → FP32 (approximate W)
# 推理计算: y ≈ x @ W_tilde.T = α·(x @ W_hat.T) + β·sum(x)
# INT 矩阵乘 x @ W_hat.T 比 FP16 GEMM 更快且省显存
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Min-Max 量化是最广泛使用的量化 baseline，在 PyTorch 中通过 `torch.quantize_per_tensor()` 实现。GPTQ、bitsandbytes 等框架将其扩展为 group-wise 版本。QA-LoRA 使用 GPTQ 的 group-wise min-max 量化作为基础量化方法，INT4 支持由 CUDA 优化的 INT4 GEMM 算子加速。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---
