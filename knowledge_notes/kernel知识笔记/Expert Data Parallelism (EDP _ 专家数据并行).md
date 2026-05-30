## Expert Data Parallelism (EDP / 专家数据并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Data Parallelism (EDP) 是 Skywork-MoE 提出的针对 MoE 训练的定制化并行策略，定义为 Size_EP = Size_TP。与 Megatron-LM Core 0.6.0 中已有的 Expert Parallelism (EP, Size_EP = Size_DP * Size_TP) 和 Expert Tensor Parallelism (ETP, Size_EP = Size_DP) 不同，EDP 的核心设计是在 Attention 层使用 Tensor Parallelism (TP)，在 MoE/FFN 层切换为 Expert Parallelism (EP)，同一数据同时穿越 TP Group 和 EP Group。Device mesh 配置：Attention weights 为 [Size_PP, Size_DP, Size_TP]，Expert weights 为 [Size_PP, Size_DP, Size_EP]。EDP 对中等 expert 数量（≤64）的 MoE 模型特别有效，能优化 gating 层 token 路由的 AllToAll 通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Skywork-MoE (1536 A800 GPUs) 的训练中 EDP 配置为 Size_PP=12, Size_DP=32, Size_TP=Size_EP=4：

```
# === Device Mesh 定义 ===
# 总 GPU 数 = PP * DP * TP = 12 * 32 * 4 = 1536
# Attention Device Mesh: [PP=12, DP=32, TP=4]
# Expert Device Mesh:    [PP=12, DP=32, EP=4]

# === 单层前向传播流程 (在 EDP 下) ===

# Phase 1: Attention Layer (TP Mesh)
# TP Group: 4 GPUs, 切分 head 维度
# 每 GPU 处理 36/4 = 9 attention heads
x = LayerNorm(x)                    # 所有 GPU 独立计算
# QKV projection: 每 GPU 计算部分 heads
q, k, v = split_heads(W_QKV @ x)    # TP 切分 head 维度
# Flash Attention: 在 TP Group 内通信
attn_out = flash_attention(q, k, v)  # 需要 TP group 内 all-reduce
attn_out = W_O @ attn_out           # TP 切分 + all-reduce
x = x + attn_out

# Phase 2: MoE Layer (EP Mesh, Mesh 切换)
# EP Group: 4 GPUs, 每 GPU 持有 16/4 = 4 个完整 expert
x = LayerNorm(x)
# Gate: 每 GPU 独立计算 (gate 参数在所有 EP GPU 上复制)
z = W_gate @ x + b_gate
z_tilde = (z - mean(z)) / std(z)    # Gating Logit Normalization
g = softmax(z_tilde)
E_i = topk(g, k=2)

# Token Dispatch: AllToAll 在 4 EP GPUs 间
# 将 token 发送到持有目标 expert 的 GPU
tokens_dispatched = all_to_all(tokens, routing_map)

# Expert FFN 计算: 每 GPU 独立计算其持有的 4 个 expert
For each expert j on this GPU:
    expert_out[j] = SwiGLU_FFN(tokens_for_expert_j)

# Token Combine: AllToAll 将 expert 输出送回原 GPU
tokens_combined = all_to_all(expert_outputs, reverse_routing_map)

# Weighted sum
y = weighted_combine(tokens_combined, g[E_i])
x = x + y
```

EDP 的核心优势：EP 组大小 = TP 组大小，使得 Attention 和 Expert 阶段的通信模式协调，避免了 ETP 中 AllToAll 随 TP 增大而迅速膨胀的问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 Skywork-Megatron 框架（基于 Megatron-LM 23.06），需要在训练框架中支持动态 device mesh 切换。具体实现要点：(1) 在 Attention 层和 Expert 层之间切换通信组（TP group ↔ EP group）；(2) 确保 expert 数量 ≥ Size_EP（才有足够的 expert 分配给每个 EP rank）；(3) Gate 参数在所有 EP ranks 上复制（非分布式）。适用场景：expert 数量 ≤ 64 的 MoE 模型训练，在通信开销和 GPU 利用率之间取得最优平衡。

涉及论文标题：
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models
