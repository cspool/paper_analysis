## Fully Differentiable MoE (Expert Merging in Parameter Space)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fully Differentiable MoE (Expert Merging in Parameter Space) 是一种完全不依赖离散路由决策的混合专家架构。传统稀疏 MoE 使用 top-k 离散选择（argmax/top-k）决定每个 token 激活哪些专家，路由决策不可微，需要辅助负载均衡损失和复杂的分配算法。相反，Fully Differentiable MoE 在参数空间对所有专家进行软合并（soft merging）：给定路由权重 e_i = Softmax(R(h))，计算所有专家参数的加权平均 θ̄ = Σ_i e_i · θ_i，然后用合并后的 FFN 处理输入 o = FFN(h; θ̄)。整个过程端到端可微，梯度通过合并操作和路由网络全程回传，无需辅助损失。

该方法首先由 SMEAR (Muqeeth et al., 2023) 提出，在 BERT 编码器的文本分类下游微调中验证。Lory (Zhong et al., 2024, COLM) 首次将该架构扩展到自回归语言模型预训练。与 SMEAR 使用池化表示对整个输入序列做一次路由不同，Lory 设计了 causal segment routing 以保留自回归因果性。

关键优势：(1) 无需离散路由，端到端梯度回传；(2) 无需负载均衡辅助损失，减少超参数调优；(3) 合并后的 FFN 在推理时等价于单个 Dense FFN，推理效率与 Dense 模型相同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === 参数空间专家合并（Lory 的 moe_ffn 核心） ===
# 输入: seg_x (B*N, T, d), e (B*N, E) 路由权重
# experts: E 个 FFN, 每个 θ_i = (W_gate_i, W_up_i, W_down_i)

# 步骤1: 在参数空间合并所有专家
merged_W_gate = sum_{i=1}^{E} e[:, i] · W_gate_i  # shape: (d, d_ffn)
merged_W_up   = sum_{i=1}^{E} e[:, i] · W_up_i    # shape: (d, d_ffn)
merged_W_down = sum_{i=1}^{E} e[:, i] · W_down_i  # shape: (d_ffn, d)

# 步骤2: 用合并后的 FFN 处理输入（SwiGLU）
gate_out = SiLU(seg_x @ merged_W_gate)   # (B*N, T, d_ffn)
up_out   = seg_x @ merged_W_up            # (B*N, T, d_ffn)
output   = (gate_out ⊙ up_out) @ merged_W_down  # (B*N, T, d)
```

与稀疏 MoE 的关键张量流对比：

**稀疏 MoE (top-k 路由)**：
1. Router: h → W_r·h → softmax → top-k → gate_weights, expert_indices
2. Token dispatch: 将每个 token 路由到选中的 k 个 expert（通过 all-to-all 通信）
3. Expert compute: 每个 expert 独立计算 FFN(h; θ_i)
4. Token combine: 将 expert 输出加权聚合（通过 all-to-all 通信）

**Fully Differentiable MoE (参数合并)**：
1. Router: h̄ → W_r·h̄ → softmax → routing weights e (E-dim)
2. Parameter merge: θ̄ = Σ_i e_i · θ_i（参数空间操作，无需 token dispatch）
3. Merged FFN compute: FFN(h; θ̄)（等价于单次 Dense FFN GEMM）

关键区别：稀疏 MoE 在激活空间路由（dispatch-combine tokens），Fully Differentiable MoE 在参数空间路由（merge parameters）。合并操作的计算开销为 O(E · d · d_ffn) per merge，若每 T 个 token 合并一次，则额外 FLOPs 为 E/T × (FFN FLOPs)，对 T=256, E=32 约 12.5%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键：
- **合并操作**：纯 PyTorch tensor ops，无需自定义 CUDA kernel。加权求和在参数张量上执行，然后调用标准 cuBLAS GEMM。
- **Expert 架构**：每个 expert 与 Dense FFN 结构完全相同（SwiGLU: W_gate + W_up + W_down），因此合并后的 FFN 也是标准 SwiGLU FFN，仅权重不同。
- **分布式训练**：Lory 使用 data parallelism + ZeRO。但专家参数量大时，可通过 expert-wise model parallelism 按 hidden dim 分片所有专家到不同设备（Section 6）。
- **推理**：Prompt-only routing——用 prompt 的平均隐藏表示计算路由权重，合并 FFN，后续所有 token 使用合并后的 FFN 生成。推理与 Dense 模型完全相同（无额外通信或计算开销）。
- **转换到稀疏推理**：Lory 模型可微调为 hard-decision routing（top-k），在推理时恢复稀疏激活以节省 GPU 内存（但论文未实现）。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training
