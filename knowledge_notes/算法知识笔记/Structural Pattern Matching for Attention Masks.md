## Structural Pattern Matching for Attention Masks

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Structural Pattern Matching 是 DAM 从 true mask 中识别标准化 attention 结构的方法。模式池 $\mathcal{P}$ 含对角线模式 $P_{\text{diag},r}$（$j=i-r$）和垂直模式 $P_{\text{vert},c}$（$j=c, i \geq c$），共 2L 个候选。匹配分数 $\gamma_k = \frac{\sum M \cdot P_k}{\sum P_k}$，$\gamma_k \geq \mu$（μ=0.8）则模式被匹配。匹配到的模式可直接外推至任意长度，使 DAM 处理超 PCL 序列无需重算 full attention。

从算法pipeline角度拆解术语：

```
// 输入: true_mask M[L×L], threshold μ=0.8
// 输出: extended mask M_ext[S×S] for any S

// Pattern pool: 2L patterns total
P = {P_diag,r: p[i,j]=1 iff j=i-r, r=0..L-1}
  ∪ {P_vert,c: p[i,j]=1 iff j=c and i≥c, c=0..L-1}

for each P_k in P:
    γ_k = (M ⊙ P_k).sum() / P_k.sum()
    if γ_k >= μ: matched.append(P_k)

// extrapolate matched patterns to length S
M_ext = sum(P_k_extrapolated(S) for P_k in matched)
M_ext = (M_ext >= 1).astype(int)  // binarize
```

术语一般如何实现？如何使用？

仅使用对角线和垂直两种模式（基于 LLaMA 3.2 3B attention 观察）。μ=0.8 在 0.7~1.0 范围内表现鲁棒。模式定义使外推极其简单——对角线条件 $j=i-r$ 和垂直条件 $j=c, i \geq c$ 对任意长度均成立。模式池可扩展（增加水平条带、块模式等）以提升精度。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration
