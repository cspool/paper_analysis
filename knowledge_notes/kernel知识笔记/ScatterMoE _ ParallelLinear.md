## ScatterMoE / ParallelLinear

术语解释
由 Tan et al. (Mila, 2023) 提出，通过 ParallelLinear 模块执行分散组的并行线性运算，避免 MegaBlocks 的 scatter-to-group 数据拷贝，中间表示保持为 PyTorch-native tensor。

术语是什么？
ParallelLinear 在不先将 token 拷贝到连续 buffer 的情况下直接执行分组矩阵运算。中间表示（如 hidden states）保持为标准 PyTorch tensor，便于扩展到非 FFN 专家模块。

从kernel调度角度拆解术语。
```
def parallellinear_forward(x, weights, group_indices):
    # x: tokens sorted by expert but not contiguous
    # group_indices: list of (start, end) per expert
    y = zeros(total_tokens, d_out)
    for expert_id, (start, end) in enumerate(group_indices):
        if start == end: continue
        y[start:end] = x[start:end] @ weights[expert_id].T
    return y
```

术语一般如何实现？如何使用？
- 开源：https://github.com/shawntan/scattermoe
- 基于 PyTorch + CUDA batched/strided GEMM
- GitHub Stars (2024.6): 140

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

**DS-MoE 中的使用**：DS-MoE (Pan et al., 2024) 在推理阶段使用 SimpleMoE 的 ParallelLinear 操作进行 MLP 层的稀疏推理。训练阶段使用 dense computation（所有 expert 全激活），无需 ParallelLinear。推理时采用混合策略：MLP 层（sparsity 高，active ratio <30-40%）使用 ParallelLinear 进行 sparse expert computation；Attention 层（sparsity 低，active ratio >60%，sparse overhead > dense benefit）使用 torch.nn dense 计算。DS-MoE 的 expert sampling 支持三种策略：Threshold（per-token 自适应选择超阈值 experts）、TopK（固定 K）、Threshold-TopK（先统计 batch 内平均 expert 数再统一 K，兼顾自适应和 batch 效率）。

---
