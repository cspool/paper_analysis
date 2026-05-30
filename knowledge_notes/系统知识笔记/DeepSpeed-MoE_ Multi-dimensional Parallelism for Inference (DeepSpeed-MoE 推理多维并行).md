## DeepSpeed-MoE: Multi-dimensional Parallelism for Inference (DeepSpeed-MoE 推理多维并行)

术语解释
DeepSpeed-MoE 推理系统是 MoE 推理的多 GPU 并行方案，组合 Expert Parallelism、Expert-Slicing、Tensor-Slicing 和 Data Parallelism 四种并行维度，针对 MoE 模型不同组成部分（expert vs non-expert ）使用不同策略，最小化每设备的 critical data path。

术语是什么？
DeepSpeed-MoE 推理的核心设计目标：将 MoE 推理性能推向"最佳情况"——每个 token 仅激活单个 expert，等效 base dense 模型参数远小于总参数量。

四种并行维度和作用：
1. **Expert Parallelism**：将不同 expert 分布到不同 GPU，token 按 gating 结果路由。EP=E（=expert 数）时，每 GPU 仅 1 expert，每 token critical path = base dense size。
2. **Expert-Slicing**：对单个 expert 内部参数进行 tensor-slicing 切分（行/列切分），当可用 GPU 数超过 expert 数时，可进一步减少每 GPU 负载。
3. **Tensor-Slicing**（节点内）：对 non-expert 参数（Attention, Embedding，固定 MLP in PR-MoE）进行模型并行，利用节点内 NVLink 高带宽。
4. **Data Parallelism**（跨节点）：对 non-expert 参数创建多副本处理不同 batch，跨节点无通信开销。

从系统架构角度拆解术语：
```
# DeepSpeed-MoE 推理：单 MoE Transformer 层（1.3B+MoE-128, 128 GPUs, EP=128, TP=8）

=== Non-expert Partition (Attention + LayerNorm) ===
# TP=8 切分，仅限节点内（8 GPUs per node）
# 每 GPU: Attention params / 8
Q_i, K_i, V_i = linear_projections(x_i)    # sharded QKV
All-Reduce(Q_i, K_i, V_i)                  # 仅限 TP group 内
attn_out = Attention(Q, K, V)              # FlashAttention

=== Expert Partition (MoE) ===
# EP=128 分布，1 expert per GPU
# Gating（在持有 token 的 GPU 执行）:
expert_id[t] = argmax(gate_logits[t])       # Top-1 expert
# All-to-All dispatch (Parallelism-Coordinated):
alltoall_subset(tokens, same_tp_rank_GPUs)  # 仅限同 TP rank 的 16 GPUs
# Expert computation:
expert_out = expert_ffn(tokens)             # 单个 expert FFN
# All-to-All combine:
alltoall_subset(expert_out, same_tp_rank_GPUs)
# AllGather within TP group (TP gpus need full data):
allgather(expert_out, tp_group)

# Critical data path per GPU: 
# Non-expert: base_params / TP = 1.3B / 8 = 0.16B params loaded per GPU
# Expert: 1 expert params (EP=128) ≈ (52B-1.3B) / 128 ≈ 0.4B params
# Much smaller than quality-equivalent 6.7B dense model loaded on 1 GPU
```

术语一般如何实现？如何使用？
- 开源：https://github.com/microsoft/DeepSpeed
- 通过 DeepSpeed InferenceEngine API 调用，自动选择最优并行配置
- Expert-Slicing 作为额外维度：W_i 列切分 + W_o 行切分（与 MoEShard 的切分策略类似但目的不同）

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale
