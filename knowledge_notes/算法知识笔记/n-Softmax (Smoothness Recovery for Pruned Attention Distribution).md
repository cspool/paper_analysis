## n-Softmax (Smoothness Recovery for Pruned Attention Distribution)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

n-Softmax 是 CSP 提出的用于恢复 KV Cache 剪枝后注意力分布平滑性的技术。剪枝后 softmax 的分母从 Σ_{j∈I^+ ∪ I^-} e^{O_j} 变为 Σ_{j∈I^+} e^{O_j}，导致注意力分数被放大、分布变尖锐。n-Softmax 在分母中引入偏置 n：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})，相当于添加"虚拟 token"的贡献来模拟被剪枝 token 的归一化效应，恢复原始平滑性。论文固定 n=1。与标准 softmax（Σ A_i = 1）不同，n-Softmax 是放松归一化（Σ A_i < 1），额外的概率质量被 n 吸收。在 CSP 算法中，n-Softmax 作为 attention score 计算的第一步，为后续的 Cross-Self 分解和 top-K 选择提供更好的分数基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 标准 softmax（剪枝前）
A_i = e^{O_i} / Σ_{j=1}^{L} e^{O_j}          // 完整分母，平滑分布

// 标准 softmax（剪枝后 — 问题所在）
A_i = e^{O_i} / Σ_{j∈I^+} e^{O_j}            // 分母变小 → A_i 变大 → 分布尖锐

// n-Softmax（解决方案）
A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})      // n=1, 恢复平滑性

// CSP 整体流程中的使用
A = n-Softmax(Q @ K^T / sqrt(d))             // attention logits → smoothed weights
M^s, M^c = CrossSelfDecomposeAndSelect(A)     // 后续 Cross-Self 分解
```

**消融效果**：在 ALFRED 数据集上，n-Softmax 配合 Cross-Self 分解带来一致且轻微的性能提升，在需要时间连贯性和细粒度特征保留的任务上尤为有效。

术语一般如何实现？如何使用？

实现仅需在 softmax 分母中加 n（一行代码修改）。n=1 在所有实验中使用。不独立使用，必须与 Cross-Self Attention Decomposition 组合。代码开源：https://github.com/TerryPei/CSP。

涉及论文标题：
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

---
