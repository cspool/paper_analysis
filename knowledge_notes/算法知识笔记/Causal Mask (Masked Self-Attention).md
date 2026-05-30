## Causal Mask (Masked Self-Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Causal Mask 是 Transformer Decoder 中自注意力操作的遮蔽机制。它在 Attention Score 矩阵 S = QK^T / sqrt(d_k) 上施加一个上三角为 -inf（或极小值）的 mask，使得 Softmax 后对应位置概率为 0，实现"当前 token 只能看到它自身和之前生成的 token，不能看到未来 token"的自回归约束。

数学形式：$S'_{i,j} = S_{i,j} + M_{i,j}$，其中 $M_{i,j} = -\infty$ if $i < j$ else $0$。Softmax 后 $\text{softmax}(-\infty) = 0$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**带 Causal Mask 的 Self-Attention**：

```
Q = x @ W_Q  // [1, N, d_k] 或 [N, d_k]
K = x @ W_K  // [1, N, d_k]
V = x @ W_V  // [1, N, d_v]

scores = Q @ K^T / sqrt(d_k)  // [N, N]
// 应用 Causal Mask
for i in 0..N:
    for j in i+1..N:
        scores[i][j] = -inf  // 上三角置 -inf

attn = softmax(scores)  // 下三角非零，上三角为 0
output = attn @ V        // [N, d_v]
```

**Causal Mask 对 A2S 的影响**：
因上三角为 0，第 k 个 token 在第 q 步（q < k）时 $S_{q,k} = 0$，只有在第 k 步及之后才产生非零分数。这导致：
- Token 1：累积 N 次
- Token 2：累积 N-1 次
- Token k：累积 N-k+1 次

A2S 值天然按 token 位置排序——早期 token 的累积优势掩盖了真实重要性。

术语一般如何实现？如何使用？

Causal Mask 是所有自回归 Transformer（GPT、LLaMA、OPT 等）的标准组件。在 PyTorch 中通常用 `torch.nn.Transformer.generate_square_subsequent_mask()` 或 `torch.triu()` 生成。FlashAttention 等优化 kernel 将其融入计算流程，避免显式构建 N×N mask 矩阵。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

---
