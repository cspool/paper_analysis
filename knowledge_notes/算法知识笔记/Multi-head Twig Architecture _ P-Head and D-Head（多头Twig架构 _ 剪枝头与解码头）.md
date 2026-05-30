## Multi-head Twig Architecture / P-Head and D-Head（多头Twig架构 / 剪枝头与解码头）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-head Twig Architecture 是 TwigVLM++ 对原始 TwigVLM 的核心架构改进。在原始 TwigVLM 中，twig block 的最后一个自注意力层的 attention map 同时服务于两个目的：next-token 预测（通过 AR loss 间接训练）和视觉 token 剪枝（推理时直接使用 attention scores）。这种耦合设计导致剪枝信号仅作为预测任务的副产品出现，未针对剪枝任务直接优化。TwigVLM++ 引入两个解耦的 head：(1) **D-Head (Decoding Head)**：保留标准 next-token 预测功能，复用原 twig block 的预测头；(2) **P-Head (Pruning Head)**：轻量级辅助模块，专用于计算视觉 token 重要性分数 s ∈ R^M。

P-Head 的计算 (Eq.7)：从 twig 最后一层 SA 的 Q/K 投影中提取 query vector q̃（最后 textual token 位置）和 key matrix K̃（visual token 位置），通过两个可学习的 gating 投影 G_q, G_k（Linear + nonlinear activation）调制后再计算 scaled dot-product attention，最终对各注意力头取平均：s = 1/H · Σ σ((G_q(x_q)⊙q̃)(G_k(X_k)⊙K̃)^T / √d_h)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# P-Head 重要性分数计算 (TwigVLM++)
# 输入: X^(K+T) — twig最后一层SA层输入
# H: 注意力头数, d_h: 头维度

Q = X_Wq    # ∈ R^{(M+N)×d}
K = X_Wk    # ∈ R^{(M+N)×d}

# 提取 query (最后text token位置)
q_tilde = Q[-1].reshape(H, d_h)    # ∈ R^{H×d_h}
x_q = X[-1]                        # 最后text token的hidden state

# 提取 key (visual token位置)
K_tilde = K[:M].reshape(H, M, d_h)  # ∈ R^{H×M×d_h}
X_k = X[:M]                         # visual tokens hidden states

# P-Head gating projections
gated_q = G_q(x_q).reshape(H, d_h) ⊙ q_tilde   # element-wise
gated_k = G_k(X_k).reshape(H, M, d_h) ⊙ K_tilde

# 多头的scaled dot-product attention
scores_per_head = []
for h in range(H):
    s_h = softmax(gated_q[h] @ gated_k[h].T / sqrt(d_h))
    scores_per_head.append(s_h)

s = mean(scores_per_head, dim=0)  # ∈ R^M, 归一化的token重要性
```

D-Head vs P-Head 对比：
| 特征 | D-Head | P-Head |
|------|--------|--------|
| 功能 | Next-token prediction | Visual token importance scoring |
| 输出 | Probability distribution over vocabulary | Scores s ∈ R^M |
| 训练 | L_NTP + L_PredKL | L_AttnKL + RL (Stage-2) |
| 推理使用 | SSD draft generation | TTP token pruning |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
P-Head 的实现：在 twig 最后一层 SA 之后添加两个轻量级 linear 层（G_q 和 G_k），后接非线性激活函数（如 GELU）。训练分为两个阶段：Stage-1 用蒸馏损失（L_AttnKL 监督 s 与深层 attention 对齐），Stage-2 用 GRPO-style RL 直接优化 s 以最大化剪枝后模型性能。推理时，P-Head 的输出 s 替代 attention map 用于 Eq.(8) 的 token 剪枝。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
