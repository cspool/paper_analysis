## Megablocks

术语是什么？
Megablocks 是由 Databricks（Gale et al., 2023）开发的高效稀疏 Mixture-of-Experts 训练库，核心创新是使用 block-sparse matrix multiplication 实现 dropless MoE routing——无需像传统 MoE 实现那样因 expert capacity 限制而丢弃 token。Megablocks 通过将 token-to-expert dispatch 问题映射为 block-sparse 矩阵乘法，利用专门的 CUDA kernel 实现对任意 per-expert token 分配的高效计算。

从编译框架角度拆解术语：
Megablocks 作为 MoE 训练编译/计算框架的工作流程：

1. **Router 计算（PyTorch 前端）**：标准的 MoE gating 计算，输出每个 token 对所有专家的 logits。
2. **Token-to-Expert Assignment**：Top-K selection 后，Megablocks 将 token 按目标 expert 重新排序（permute），生成 block-sparse 格式的索引和偏移量（类似 CSR/CSC 稀疏格式）。
3. **Block-Sparse Matrix Multiplication（CUDA kernel）**：
   - 输入被划分为固定大小的 block（如 128×128）
   - 只有包含有效 token-expert 对的 block 参与计算
   - 使用定制 CUDA kernel（基于 CUTLASS）高效执行 block-sparse GEMM
   - 支持混合精度（FP16/BF16）训练
4. **Token Un-permute**：将各 expert 输出按原始 token 顺序重新排列，传递给后续层。
5. **输出**：与标准 dense FFN 相同格式的输出，无缝集成到 Transformer block 中。

论文使用 Megablocks 训练所有 dropless MoE 模型，结合 Router Z-Loss 和 QK-Normalization 实现稳定的大规模 MoE 训练。

术语一般如何实现？如何使用？
Megablocks 通过 PyTorch 扩展实现，核心计算由 C++/CUDA 提供。典型使用方式：
```python
from megablocks.layers import moe

# 定义 MoE layer
moe_layer = moe.MoE(
    hidden_size=d_model,
    ffn_hidden_size=4 * d_model,
    num_experts=E_total,
    top_k=E_active,
    activation=gelu,  # 或 swiglu
)

# 前向传播（dropless）
output, router_logits = moe_layer(hidden_states)
```

Megablocks 的关键依赖：(1) block-sparse CUDA kernel 需要较新的 NVIDIA GPU（A100/H100）；(2) 使用 `torch.bfloat16` 训练以匹配 kernel 精度要求；(3) 需要配合 load balancing loss 防止 expert 使用不均（因无 capacity 约束）。论文在 hyperparameter search 中 sweep 了 load balancing factor ∈ {0.02, 0.05}。

涉及论文标题：
- Parameters vs FLOPs Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models
- TurboMoE Enhancing MoE Model Training with Smart Kernel-Fusion and Data Transformation
