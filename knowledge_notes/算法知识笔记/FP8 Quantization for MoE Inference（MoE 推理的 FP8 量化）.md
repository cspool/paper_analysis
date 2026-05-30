## FP8 Quantization for MoE Inference（MoE 推理的 FP8 量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP8（8-bit Floating Point）量化是一种将神经网络模型的权重和激活从 FP16/BF16（16-bit）降低到 8-bit 浮点精度的压缩技术。与 INT8 量化不同，FP8 保留了浮点数的指数位（exponent bits），因此具有更大的动态范围（dynamic range），特别适合需要同时表示极大值和极小值梯度的 LLM 推理场景。FP8 格式有两种常见变体：E4M3（4-bit exponent + 3-bit mantissa，精度更高，适合前向传播的权重和激活）和 E5M2（5-bit exponent + 2-bit mantissa，动态范围更大，适合梯度）。NVIDIA H100 GPU 的第四代 Tensor Core 原生支持 FP8 计算，FP8 Tensor Core 的峰值算力是 FP16/BF16 的 2 倍（H100 SXM5 的 FP8 峰值达 1979 TFLOPS vs FP16 的 989.5 TFLOPS）。在 MoE 推理场景中，FP8 量化可显著减少显存占用（将 expert 权重参数从 FP16 的 2 bytes/param 降至 1 byte/param），使更多 expert 参数可同时驻留在 GPU 显存中，或支持更大的 batch size。MoE-Inference-Bench 使用 GPTQ 和 AWQ 等 post-training quantization 方法实现 FP8 量化，并通过 vLLM 在 H100 上评估。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FP8 量化 MoE 推理的算法 pipeline（以 Mixtral-8x7B on H100 + vLLM with FP8 为例）：

```
# MoE Layer with FP8 Quantization (pseudocode, per MoE layer)
# Input: hidden_states [batch_size, seq_len, hidden_dim=4096], dtype=FP16

# Step 1: Router (kept in FP16 for routing accuracy)
router_logits = fp16_matmul(hidden_states, W_gate_fp16)  # [B, S, num_experts=8]
gate_weights, topk_indices = topk(softmax(router_logits), k=2)

# Step 2: FP8 Quantized Expert FFN (quantized weights loaded as FP8)
for expert_id in range(8):
    tokens_for_expert = hidden_states[topk_indices == expert_id]  # FP16 input

    # FP8 matmul: input (FP16) × weight (FP8) → accumulate in FP32 → output (FP16)
    # H100 Tensor Core: FP8 E4M3 weights, FP16 inputs auto-promoted
    gate_out = fp8_matmul(tokens_for_expert, W_gate_fp8[expert_id])  # [n, 14336]
    gate_out = silu(gate_out)

    up_out   = fp8_matmul(tokens_for_expert, W_up_fp8[expert_id])    # [n, 14336]
    expert_hidden = gate_out * up_out                                  # element-wise

    expert_out = fp8_matmul(expert_hidden, W_down_fp8[expert_id])    # [n, 4096]

    # Weighted sum accumulation (FP32 for numerical stability)
    output[topk_indices == expert_id] += gate_weights[expert_id] * expert_out
```

MoE-Inference-Bench 的 FP8 vs FP16 关键发现（Section 6.1）：(a) FP8 在 batch size=64 时提供 25-30% 更高吞吐量（batch size 越大，FP8 优势越明显，因为显存节省允许更大的有效 batch）；(b) 在不同 sequence length 下 FP8 保持 20-25% 吞吐量优势（鲁棒于 context length 变化）；(c) FP8 在 compute-bound 和 memory-bound 两种场景下均有效——compute-bound 场景受益于 2× Tensor Core 算力，memory-bound 场景受益于减半的权重显存带宽需求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FP8 量化在 MoE 推理中的实现方法：
- **Post-Training Quantization (PTQ)**：不需要重新训练，通过对已训练的 FP16 模型应用量化算法（GPTQ/AWQ）直接转换为 FP8。GPTQ 基于逐层 Hessian 矩阵的最小二乘优化，AWQ 基于 activation 感知的权重重要性缩放。MoE-Inference-Bench 使用这些方法。
- **框架支持**：vLLM 通过 PyTorch 的 `torch.fp8` 支持和自定义 FP8 kernel 进行 MoE 的 FP8 推理。TensorRT-LLM 支持 MoE 模型的 FP8 推理（NVIDIA blog 报告 Mixtral 8x7B 在 2×H100 上 FP8 streaming 达 38.4 req/s）。
- **硬件要求**：需要 H100 (SM90) 或更新架构（FP8 Tensor Core 支持）。A100 及以下不支持 FP8 Tensor Core。
- **精度保持**：MoE-Inference-Bench 中 FP8 量化不显著影响模型质量（Section 8 的准确率实验在 FP16 下运行），论文未报告 FP8 下的准确率对比。一般经验：FP8 E4M3 对推理精度影响极小（<0.1% perplexity 退化）。
- 局限：Router 和 attention 层通常保持 FP16/BF16（量化 router 会显著影响 routing 决策精度）；需要 per-tensor 或 per-channel 的 scaling factor 管理 FP8 的有限动态范围。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
