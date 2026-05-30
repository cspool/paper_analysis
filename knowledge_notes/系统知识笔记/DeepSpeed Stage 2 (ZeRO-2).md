## DeepSpeed Stage 2 (ZeRO-2)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

DeepSpeed Stage 2 (ZeRO-2) 是 Microsoft DeepSpeed 框架的分布式训练优化级别，通过分片优化器状态和梯度来减少每个 GPU 显存占用。与 ZeRO-1 仅分片优化器状态（fp32 master params + momentum + variance = 12 bytes/param）不同，ZeRO-2 额外分片梯度（2 bytes/param in fp16），将梯度通信从 AllReduce 变为 Reduce-Scatter。ZeRO-2 不复制权重参数，前向后向仍需 AllGather 收集完整参数。通常可实现 8× 以上显存节省（8 GPU），使 7B-70B 模型可在适度 GPU 集群上训练。

从系统架构角度拆解术语。

KV-Distill 在 8× NVIDIA A100 80GB 集群上使用 DeepSpeed Stage 2 训练。由于仅训练 150M LoRA 参数（base model frozen），ZeRO-2 主要优化 optimizer states 和 gradients 的显存分配。训练配置：bf16、AdamW、batch 32。LLAMA-2/3 7B-8B、MISTRAL 7B、GEMMA-2 9B 3 天收敛，GEMMA-2 27B 4 天。

术语一般如何实现？如何使用？

配置文件：`{"zero_optimization": {"stage": 2}}`，`deepspeed.initialize()` 包装模型和优化器，或 HuggingFace `--deepspeed ds_config.json`。ZeRO-2 在 backward() 后 hook 梯度流做 Reduce-Scatter。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

---
