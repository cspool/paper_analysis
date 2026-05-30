## Expert Generalization and Specialization in Multimodal MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Generalization and Specialization 是 Flex-MoE 提出的两阶段 SMoE 训练策略，用于在 missing modality 场景下让每个 expert 同时具备通用知识和专有知识。灵感来自课程学习——先用"简单"样本（全模态）学习通用知识，再用"困难"样本（部分模态）学习专有知识。

- **Generalization 阶段（warm-up epochs）**：仅使用全模态样本（all modalities observed），G-Router 执行标准 top-k gating + load/importance balancing loss，让所有 expert 学习从完整多模态信息中提取的通用知识。
- **Specialization 阶段（剩余 epochs）**：使用所有 modality combination 的样本，S-Router 通过 cross-entropy loss 将 top-1 gate 强制绑定到目标 modality combination expert index，其余 top-(k-1) expert 继续做 load/importance balancing。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === Phase 1: Expert Generalization (warm-up epochs) ===
# 输入: 仅全模态样本 batch (all 4 modalities observed)
for sample_i in full_modality_batch:
    h_i = concat([e_i^I, e_i^C, e_i^B, e_i^G])  # 所有模态真实编码
    gate_logits = g(h_i)                         # G-Router (1-2 layer MLP)
    gate_vals = TopK(softmax(gate_logits), k)    # k=4 for ADNI
    y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)
    L = L_CE(y_i, label) + 0.01 * L_balance     # 全 expert 参与 balancing

# === Phase 2: Expert Specialization (remaining epochs) ===
# 输入: 任意 modality combination 的样本
for sample_i in batch:  # batch 按可用模态数降序排列
    h_i = flex_moe_encode(sample_i)              # 含 missing modality bank
    gate_logits = g(h_i)                         # S-Router
    top1_pred = argmax(gate_logits)
    target_exp = MC_index(observed_modalities(i))
    L_ce = -sum_j one_hot(MC(x_j)) * log(softmax(gate_logits))  # 绑定 top-1
    gate_vals = TopK(softmax(gate_logits), k)
    y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)
    # L_balance 仅计算 E \ {e_top1} 的 expert
    L = L_CE(y_i, label) + 0.01 * (L_balance + L_ce)
```

Flex-MoE 在 ADNI 数据集上验证：去除 Expert Specialization 后 ACC 从 66.11 降至 62.75；同时去除 ES+EG 后降至 62.49。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 样本排序——训练开始前按可用模态数降序排列所有样本，warm-up 阶段自然只包含全模态样本；(2) Expert index 分配——每种 modality combination 对应固定 expert index（如 "IGCB"=0, "IGC"=1, ..., "B"=14），剩余 index 为 buffer expert；(3) S-Router 的 cross-entropy loss 直接作用于 gate logits 的 softmax，不打断梯度流；(4) warm-up epochs 后使用 shuffled 样本增强泛化性；(5) top-k 选择在 specialization 阶段偏大（k=4 for ADNI），因为 top-1 已固定，剩余 3 个 expert 提供跨模态组合的交互。

涉及论文标题：
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
