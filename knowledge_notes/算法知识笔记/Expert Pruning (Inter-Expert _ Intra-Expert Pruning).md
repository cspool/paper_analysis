## Expert Pruning (Inter-Expert / Intra-Expert Pruning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Pruning 是 MoE 模型的后训练压缩方法，通过删除冗余 expert 或其内部维度来减少模型大小。分为两类：(1) **Inter-Expert Pruning**（如 NAEE）：删除整层中不重要的 expert，保留的 expert 继续被 router 选择，但 top-k 不变——导致剩余 expert 需处理更多 token，造成负载不均衡；(2) **Intra-Expert Pruning**（如 MoE-I²）：缩减每个 expert 内部 FFN 的 intermediate 维度（如从 14336 缩减到 10752），保留所有 expert 但每个 expert 计算量减少。两种方法都依赖 calibration 数据集评估 expert 重要性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Inter-Expert Pruning Pipeline (NAEE-style)
# 目标: 每层删除 p% 的 expert

# Step 1: Calibration-based importance scoring
for layer in moe_layers:
    for expert in layer.experts:
        # 在 calibration set 上计算移除该 expert 后的 loss 增加量
        importance[expert] = ΔLoss when removing expert

# Step 2: 逐层删除最低 importance 的 experts
for layer in moe_layers:
    prune_count = int(num_experts * p)
    prune_experts = bottom_k(importance[layer], prune_count)
    remove_from_model(prune_experts)  # 永久删除

# Step 3: 推理时仍用原始 top-k
# 问题: 剩余 expert 数量减少，但 k 不变 → 每个 expert 处理更多 token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NAEE (Lu et al., 2024) 从 calibration set 计算 expert 对输出的贡献度来排序，并额外提出 token-aware dynamic expert skipping，可在推理时跳过某些 token 的 expert 计算，但仅支持 top-k=2。MoE-I² (Yang et al., 2024) 进一步结合 inter-expert pruning + intra-expert low-rank decomposition (SVD)，用遗传算法搜索最优的逐层剪枝比例。LExI 实验表明，vLLM 上 pruning 的吞吐量提升有限甚至退化——因为稀疏 routing 不变，剩余 expert 负载增加导致长尾 latency。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference
