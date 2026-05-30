## Attention Decay under MoE Expert Pruning

术语是什么？
Attention Decay（注意力衰减）是 MC-MoE 发现的一种 expert 剪枝引起的级联效应：在 MoE layer L 中对某 token 的 expert 进行剪枝后，该 token 的 hidden state 表示质量下降，进而在下一层（L+1）的 self-attention 中，该 token 无法吸引其他 token 的注意力（attention score 降低），导致关键上下文信息丢失。具体表现为：未剪枝时 attention map 中有明显垂直高亮列（其他 token 高度关注此 token），weight-only pruning 后该列 score 显著降低。这解释了为什么仅用 routing weight 做剪枝会在 15% 剪枝率下造成 ~10% LM-Eval 精度损失。保护 top 2% 重要 token 即可有效缓解此效应。

从算法pipeline角度拆解术语：
```
Block L: token t_j 的次 expert 被 weight-only pruning 剪枝
  → h_j^L = w_0·E_0(t_j) (vs 正常的 w_0·E_0 + w_1·E_1)
  → h_j^L 信息量降低

Block L+1:
  → Q_j = W_Q @ h_j^L, K_j = W_K @ h_j^L  // 受污染的表征
  → A[:,j] = softmax(Q @ K^T/√d_k)[:,j]   // attention score 降低
  → 其他 token 对 t_j 关注度下降，信息传递受阻
```

术语一般如何实现？如何使用？
- 检测：对比剪枝前后 attention map 特定 token 列的 score 变化
- 缓解方案：(a) Token-aware protection（MC-MoE ODP）；(b) Attention-aware pruning metric（在 pruning 决策中直接考虑 attention map 影响）；(c) Layer-wise adaptive threshold
- 意义：MoE 压缩不能仅关注 expert 层面的精度损失，必须考虑 token 间注意力交互的级联效应

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
