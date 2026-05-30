## Mixture of Attention Heads (MoA)

术语解释
由 Zhang et al. (EMNLP 2022) 提出，将 MoE 思想应用于多头注意力：每 token 通过门控动态选择注意力头而非使用全部注意力头，实现 attention 层的条件计算。

术语是什么？
两套 experts（Q projection + O projection），共享门控，K 和 V 在所有 experts 间共享。仅对选中的 attention experts 计算 Q 和 O。DS-MoE 发现 Attention layers 稀疏度远低于 FFN layers（80% vs 20% active ratio），因此保持 attention experts 全激活。

```
def moa_forward(x, Wk, Wv, q_experts, o_experts, gate, K):
    K, V = x @ Wk, x @ Wv  # shared
    topk_idx = TopK(softmax(gate(x)), K)
    y = sum(gate_w[e] * softmax(x@q_e @ K.T/sqrt(d)) @ V @ o_e.T for e in topk_idx)
    return y
```

术语一般如何实现？如何使用？
- 减少 Attention 计算量（仅激活部分 heads）
- 后续工作：JetMoE, ModuleFormer 沿用此设计
- 适用于长序列场景以节省 attention 计算

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

**DS-MoE 中的 MoA 使用**：DS-MoE (Pan et al., 2024) 将 MoA 与 Grouped-Query Attention (GQA) 结合。每个 MoA expert 负责计算 N_head 个 query vectors Q_i = W_q_i @ X，其中 W_q_i ∈ R^{N_head × d_head × d_h}。K、V 在所有 expert 间共享（通过 GQA 机制共享 KV heads）。最终输出：O = Σ_{k=1..K} S_{A_k} Σ_{j=1..N_head} O_{A_k, j}，其中 O_{A_k, j} = Softmax(Q_{A_k, j} @ K^T) @ V @ W_o_j。DS-MoE 的 1B 模型使用 N_att=16, N_head=2；3B/6B 模型使用 N_att=8, N_head=4。训练阶段所有 MoA expert 全激活（dense training），推理阶段仅激活 top-K 或超阈值 expert。论文发现 Attention 层 sparsity 低于 MLP 层（active ratio >60% vs <30%），因此推理时保持 Attention 层使用 dense inference。

**BTS/BAM 中的 MoA 使用**：BAM (Zhang et al., 2024) 同时使用 MoE 和 MoA 模块，均采用 soft-routing（所有 Expert 始终激活）。MoA 的软路由输出为：
$$y_{\text{MoA}} = \sum_{i \in \mathcal{M}} q_i(x) W_{\text{attn proj}_i} (\text{Attention}_i(x))$$
其中 $q_i(x)$ 为 attention router 输出的软权重，$W_{\text{attn proj}_i}$ 为 BAM Adapters 变体中可训练的线性 adapter。BAM 在 BTS 论文中作为 Expert Upcycling baseline（全参数训练，8.4B 训练参数）。

---
