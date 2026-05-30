## Activation Checkpointing (Gradient Checkpointing / 激活检查点)

术语是什么？
Activation Checkpointing (Chen et al., 2016; Griewank & Walther, 2000) 是一种用计算换内存的训练优化技术。在 training forward pass 中，不是保存所有中间激活值（activations），而是在 backward pass 需要时从最近的 checkpoint 重新计算（recompute）。这可将内存复杂度从 O(n) 降低到 O(sqrt(n)) 或 O(log n)，但以额外的 20-30% 前向计算量为代价。在 MoE 训练中，ES-MoE 和所有 baseline (Fairseq, Tutel, ZeRO-Offload^E) 均启用 activation checkpointing，以进一步减少 GPU 显存压力。

从系统架构角度拆解：
Activation Checkpointing 在 MoE layer 中的应用：

```
# 标准训练（无 checkpointing）:
forward: store ALL activations → GPU memory: O(n_layers * activations_per_layer) 
backward: read stored activations → compute gradients

# 使用 checkpointing:
forward: 
  - Partition layers into segments (checkpoint boundaries)
  - Store activations only at segment boundaries (checkpoints)
  - Free intermediate activations (节省显存)
backward:
  - Need activation for layer L: if L is at checkpoint → use stored
  - Otherwise → recompute from nearest earlier checkpoint
  - This triggers a "mini-forward" for the recomputed segment
```

ES-MoE 与其他框架均启用此技术，使得 GPU 显存主要用于 non-expert parameters + active expert + 当前 segment 的 activations + checkpoint 边界 activations。

术语一般如何实现？如何使用？
- PyTorch: `torch.utils.checkpoint.checkpoint()` 包装 module 的 forward 函数
- Fairseq: `--checkpoint-activations` flag
- DeepSpeed: activation checkpointing 作为 ZeRO stage 的集成特性
- Megatron-Core: selective activation checkpointing（仅 recompute attention 和 FFN，保留 layer norm）

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
