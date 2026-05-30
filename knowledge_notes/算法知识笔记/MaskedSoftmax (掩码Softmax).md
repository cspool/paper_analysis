## MaskedSoftmax (掩码Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MaskedSoftmax 是 Dynamic-LLaVA 在训练阶段使用的带二值 mask 矩阵的 Softmax 变体。在端到端训练 token pruning predictor 时，标准做法是将非必要 token 的 value 设为零向量，但这会破坏自回归过程——丢弃 output text token 的 value 意味着该 token 无法用于预测下一个 token，使 language modeling loss 计算断裂。MaskedSoftmax 将 mask 应用于 attention score 矩阵（而非 value 向量），既隔离了非必要 token 的影响，又保持了完整的自回归训练结构。

公式（Eq. 7）：

$$\text{MaskedSoftmax}(\mathbb{X}_{i,j}, \mathbb{G}) = \frac{\exp(\mathbb{X}_{i,j})\mathbb{G}_{i,j}}{\sum_{k=1}^{N_l} \exp(\mathbb{X}_{i,k})\mathbb{G}_{i,k}}$$

其中 $\mathbb{X} \in \mathbb{R}^{N_l \times N_l}$ 是 QK^T/√d_k，$\mathbb{G} \in \{0,1\}^{N_l \times N_l}$ 由 predictor mask $\mathcal{M} = \mathcal{M}^I \cup \{1\}^{N^T} \cup \mathcal{M}^{OT}$ 构造，且 $\operatorname{diag}(\mathbb{G}) = 1$（每个 token 始终能 attend 到自己）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 构造 mask 矩阵
M = M^I ∪ {1}^{N^T} ∪ M^{OT}      // [N_l] 全 token 集合 mask
G = {M}^{N_l}                      // [N_l, N_l], 每行 = M
diag(G) = 1                        // 对角线强制为 1

// 修改 Attention 计算
scores = Q @ K^T / sqrt(d_k)       // [N_l, N_l]
// Causal mask 仍正常应用: scores[future] = -inf
// MaskedSoftmax: mask 矩阵通过乘法隔离非必要 token
attn_weights = MaskedSoftmax(scores, G)
// = exp(scores) * G / Σ exp(scores) * G  (element-wise)
O = attn_weights @ V
```

术语一般如何实现？如何使用？

与标准 causal attention mask 的对比：causal mask 是将未来位置设为 -∞（加法操作），MaskedSoftmax 是在 softmax 分子和分母中通过乘法引入二值 mask。两者可联合使用。Dynamic-LLaVA 训练时 causal mask 保证每个 token 仅基于前文特征做决策（与 inference 一致），MaskedSoftmax 隔离非必要 token 的 attention 影响（与 inference 时的 token 移除等价）。消融实验（Tab. 7）：w/o MaskedSoftmax 导致 VQAv2 下降 1.1%（77.8→76.7）、GQA 下降 1.5%（61.3→59.8）。

涉及论文标题：
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification
