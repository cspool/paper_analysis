## Expert Tensor Parallelism (ETP / 专家张量并行)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Tensor Parallelism (ETP) 是 Megatron-LM Core 0.6.0 中的一种 MoE 训练并行策略，定义为 Size_EP = Size_DP。与 EP（每个 expert 完整驻留在单个 GPU 上，受 expert 数量上限限制 GPU 扩展）不同，ETP 允许将单个 expert 的权重切分到多个 GPU 上（通过 Size_TP），从而突破 EP 的 GPU 扩展上限（expert 数量限制）。然而代价是 AllToAll 通信开销随 Size_TP 增大而迅速增加，因为每个 token 的 expert 输出需要在更多的 GPU 间进行集合通信。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ETP 的 device mesh 为 [Size_PP, Size_DP=Size_EP, Size_TP]：

```
# ETP 配置示例: Size_PP=8, Size_DP=Size_EP=16, Size_TP=4 → 512 GPUs
# 每个 expert 被切分到 4 个 GPU 上 (TP)
# 每个 EP group 内 4 个 GPU 共同持有 1 个完整 expert

# Expert FFN 前向 (ETP 模式下):
For each GPU in EP group:
    # Expert 权重在 TP 维度切分
    # W_gate: [d_model, d_ffn/TP] — 列切分
    # W_up:   [d_model, d_ffn/TP] — 列切分
    # W_down: [d_ffn/TP, d_model] — 行切分
    
    # 局部计算
    h1 = W_gate_partial @ x     # [batch, d_ffn/TP]
    h2 = W_up_partial @ x       # [batch, d_ffn/TP]
    h = SwiGLU(h1) * h2
    partial_out = W_down_partial @ h  # [batch, d_model]
    
    # TP group 内 all-reduce 得到完整输出
    expert_out = all_reduce(partial_out)
```

相比 EDP (Size_EP = Size_TP)：
- ETP: Size_EP = Size_DP → EP 组更大，单个 expert 分布在更多 GPU 上，AllToAll 通信量更大，但可用 GPU 不受 expert 数量限制
- EDP: Size_EP = Size_TP → EP 组更小（等于 TP 组大小），AllToAll 通信更高效，但 GPU 扩展受 expert 数量影响

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ETP 在 Megatron-LM Core 0.6.0 中实现。适用场景：expert 数量较少但需要大规模 GPU 集群训练的 MoE 模型。实际选择：当 expert 数量 ≤ 64 时，EDP 因通信效率更高而优于 ETP；当 expert 数量很大（如 128+）时，EP 可能已经足够，ETP 的 TP 切分优势不显著。

涉及论文标题：
- Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models
