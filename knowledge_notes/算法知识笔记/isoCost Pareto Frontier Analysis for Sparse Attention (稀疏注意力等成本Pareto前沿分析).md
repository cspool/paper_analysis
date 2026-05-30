## isoCost Pareto Frontier Analysis for Sparse Attention (稀疏注意力等成本Pareto前沿分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

isoCost Pareto Frontier Analysis 是 Sparse Frontier 论文引入的稀疏注意力评估方法论。核心思想：在相同计算成本下（prefilling 用 FLOPs、decoding 用 memory transfers），比较不同模型大小 × 稀疏度配置的 accuracy，识别 Pareto 最优配置——即不被任何其他配置同时在成本和性能上支配的配置。这种方法回答了一个实际问题："给定固定计算 budget，应该用大稀疏模型还是小密集模型？"

方法论关键步骤：(1) 对每个 (model_size, sparsity_level) 配置计算计算成本——prefilling 用 FLOPs 公式（含 attention/QKV投影/MLP/embedding 以及 sparse indexing overhead），decoding 用 memory transfers 公式（含 weight loading + KV cache 加载）；(2) 对每个配置在多个任务上评估 accuracy；(3) 在 cost-accuracy 空间中绘制所有配置点；(4) 识别 Pareto 前沿——不被任何其他点支配的边界点集。

Sparse Frontier 的核心发现：对于长序列（128K），只有高稀疏度配置（0.8-0.93 sparsity, i.e. 1/5-1/15 attention budget）处于 Pareto 前沿上。大稀疏模型在等成本下始终优于小密集模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# isoCost Pareto Frontier 分析流程
# Step 1: 成本计算（prefilling FLOPs 示例）
for model_size in [7B, 14B, 32B, 72B]:
    for sparsity in [0, 0.33, 0.5, 0.6, 0.7, 0.8, 0.87, 0.9, 0.93, 0.95]:
        ρ = 1 - sparsity  # attention density
        # Attention FLOPs (Eq. 3 from paper):
        FLOPs_attn = N_layers * (2*L*d*(d + 2*d_h*n_kv + d) 
                     + ρ * (2*h*L²*d_h + 3*h*L² + 2*h*L²*d_h))
        # Total prefill FLOPs:
        FLOPs_total = FLOPs_embed + FLOPs_attn + FLOPs_mlp + FLOPs_logits

# Step 2: 计算 accuracy（所有 9 个任务平均）
accuracy = mean(task_accuracy over 9 tasks for this config)

# Step 3: 绘制并识别 Pareto frontier
points = {(cost_i, acc_i) for all configs}
pareto_frontier = []
for p in points:
    if not exists q: q.cost < p.cost AND q.acc >= p.acc:
        pareto_frontier.append(p)  # p is Pareto-optimal

# Step 4: 沿着 Pareto frontier 比较效率交叉点
# "efficiency crossover" = 大稀疏模型开始优于小密集模型的点
```

术语一般如何实现？如何使用？

isoCost 分析方法的关键价值在于提供 hardware-agnostic 的效率比较框架——FLOPs 和 memory transfers 在优化实现下与 wall-clock time 高度相关，但避免了特定硬件/实现的具体延迟测量偏差。适用场景：(a) 稀疏注意力方法的理论效率对比；(b) 部署决策——选择给定 budget 的最优 (model_size, sparsity) 配置；(c) 指导未来稀疏注意力设计方向。局限性：未考虑 batch size 效应（batch_size=1 时 decode KV cache 占比低，稀疏收益小）和 memory hierarchy 效应（cache hit/miss）。

涉及论文标题：
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
