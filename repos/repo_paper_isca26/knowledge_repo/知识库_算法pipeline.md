# 知识库_算法pipeline

## Mixture-of-Experts (MoE)

术语是什么？
Mixture-of-Experts (MoE) 是一种神经网络架构，将传统 Transformer 的 Feed-Forward Network (FFN) 层拆分为多个小型 FFN（称为 expert）。每个输入 token 通过 Gate Network（门控网络）动态选择 top-k 个 expert 进行计算，而非激活全部 expert。这使模型参数量大幅增长（通过增加 expert 数量），但计算量仅线性增长于激活的 expert 数（而非总参数量），实现计算效率与模型容量的解耦。

从算法pipeline角度拆解：
MoE 层的算法 pipeline 如下：

```
# 输入: hidden_states [batch, seq_len, hidden_dim]
# Gate Network: 计算每个 token 对每个 expert 的亲和度分数
gate_logits = Linear(hidden_states, num_experts)  # [B, S, E]
# 选择 top-k experts
topk_weights, topk_indices = TopK(Softmax(gate_logits), k=topk)
# 归一化选中的权重
topk_weights = Softmax(topk_weights) / sum(topk_weights)

# 对每个选中的 expert e_i:
for i in range(topk):
    expert_idx = topk_indices[:, :, i]
    # Expert FFN: 两层 GEMM + 激活函数
    # GEMM-1: hidden_dim -> moe_intermediate_dim
    h = Activation(Linear(hidden_states, W_in[expert_idx]))
    # GEMM-2: moe_intermediate_dim -> hidden_dim  
    output_i = Linear(h, W_out[expert_idx])
    # 加权
    output_i *= topk_weights[:, :, i]

# Combine: 将 topk 个 expert 输出加权求和
final_output = sum(output_i for i in range(topk))
```

术语一般如何实现？如何使用？
- 主流 MoE 模型：DeepSeek-V3（256 experts, topk=8）、Mixtral 8x7B（8 experts, topk=2）、GPT-OSS-120B（64 experts, topk=4）、Qwen3-235B（128 experts, topk=8）。
- 门控网络通常为单层 Linear + Softmax + TopK 选择，部分实现加入负载均衡损失（load balancing loss）和辅助损失（auxiliary loss / z-loss）防止 expert 崩溃。
- Expert FFN 内部结构通常为 SwiGLU 或标准 ReLU/GELU 激活 + 两层 GEMM。
- 训练时每个 expert 可分布在不同的 GPU 上（Expert Parallelism），推理时可合并或分布部署。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch

## Top-k Gating / Gating Network

术语是什么？
MoE 中的 Gating Network（门控网络）是一个轻量级路由模块，通常为一个 Linear(hidden_dim, num_experts) 全连接层。它对每个 token 的 hidden state 产生一个分数向量，经 Softmax 归一化后选出分数最高的 k 个 expert（top-k selection），并将 token 分发到这些 expert 进行计算。选中的 expert 权重经重新归一化后用于 Combine 阶段的加权求和。

从算法pipeline角度拆解：
```
# 门控计算
gating_scores = GateLinear(token_hidden)        # shape: [num_tokens, num_experts]
gating_probs = Softmax(gating_scores, dim=-1)
# TopK 选择
topk_weights, topk_indices = torch.topk(gating_probs, k=topk, dim=-1)
# 重新归一化（可选）
topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
# 路由决策: token -> experts[topk_indices]
```

术语一般如何实现？如何使用？
- 实现为单层 Linear 层，输入为 token 的 hidden state，输出维度为 num_experts。
- topk 通常为 1-8（DeepSeek-V3 用 8，Mixtral 用 2）。更大的 topk 增加计算量但可能提升模型质量。
- 负载均衡策略：通过辅助损失鼓励均匀的 expert 利用率，避免部分 expert 过载而其他闲置（expert collapse）。
- 推理时的 token 分布与训练不同：训练近似正态分布（std≈0.032），推理近似幂律分布（α≈1.5）。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch
