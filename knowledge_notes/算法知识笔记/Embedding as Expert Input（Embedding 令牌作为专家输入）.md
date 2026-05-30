## Embedding as Expert Input（Embedding 令牌作为专家输入）

术语是什么？
Embedding as Expert Input 是 MoLE 的关键设计选择：将 routed expert 的输入从 Transformer 中间层的 hidden states h（含上下文信息的连续向量）改为 embedding 层的输出 e = Embedding(input_ids)。这一修改的根本目的是限制 expert 的输入空间——从无限连续的 R^d 收缩为有限离散集 |V|（vocabulary size），从而使 expert 可以被重参数化为 Lookup Table。直接代价是 expert 丧失了直接访问上下文信息的能力（补偿：shared expert 和 attention 层仍处理中间特征）。

从算法pipeline角度拆解术语：
```
# 标准 MoE (expert 接受中间特征)
h = Attention(LN(x)) + x                # 中间特征，含上下文
g = SoftMax(Router(h))                  # Routing 基于上下文
expert_out = FFN_j(h)                   # Expert 看到完整上下文表示

# MoLE (expert 接受 embedding tokens)
e = Embedding(input_ids)                # embedding 输出，不含上下文
h = Attention(LN(x)) + x                # 中间特征仍用于 Attention + Router
g = SoftMax(Router(h))                  # Routing 基于上下文（context-aware）
expert_in = RMSNorm(e)                  # Expert 只看到词级别信息
expert_out = FFN_j(expert_in)           # 对纯词 embedding 做变换
output = Σ_j g_j * expert_out + FFN_shared(h) + h
```

关键：Router 的输入仍是中间特征 h（含上下文），因此 routing 决策本身是 context-aware 的。MoLE ablation 显示此修改仅造成 0.7 point AVG 下降（Table 7），但带来 expert 可重参数化 + 全激活的收益。

术语一般如何实现？如何使用？
- PyTorch 实现：`embedding_states = self.expert_layernorm(embedding_states)` → `routed_output = torch.stack([expert(embedding_states) for expert in self.routed_expert], dim=2)`
- expert_layernorm（RMSNorm）确保 embedding 输入的 scale 与中间特征一致
- 适用条件：必须存在离散输入空间（如 language token vocabulary），|V| 不能过大（否则 LUT 存储不可接受）
- 不适用：视觉模型（连续像素输入）、语音模型（连续频谱输入）等无固定离散 vocabulary 的模态

涉及论文标题：
- Mixture of Lookup Experts
