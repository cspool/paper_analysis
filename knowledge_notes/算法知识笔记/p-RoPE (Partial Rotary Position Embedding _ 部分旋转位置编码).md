## p-RoPE (Partial Rotary Position Embedding / 部分旋转位置编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

p-RoPE (Barbero et al., 2024b) 是 RoPE 的一种变体，通过降低 effective base $\theta_{\text{eff}}$（如 1024 vs 标准 10000）来排除低频/高波长成分。RoPE 的频率为 $\theta_i = \theta_{\text{base}}^{-2i/d}$，高 $i$ 对应低频/长波长（编码远距离位置）。这些低频成分在长上下文泛化时可能有害：其周期超过训练时最大序列长度，推理时遇到未见过的相位。p-RoPE 通过降低 $\theta_{\text{base}}$ 将所有波长限制在更短范围。

从算法pipeline角度拆解术语：

p-RoPE 的实现仅需修改 RoPE 的 $\theta_{\text{base}}$。对 $d=128$：RoPE 最大波长 $\approx 56000$ tokens，p-RoPE ($\theta=1024$) 最大波长 $\approx 5730$ tokens。

**与 Scale-invariant Attention 的关系**：论文发现 scale-invariant RoPE 在长上下文泛化时不如 scale-invariant p-RoPE。假设：RoPE 的低频成分会干扰位置依赖的 logit 变换 $a_t, m_t$，而 p-RoPE 通过移除低频成分消除此冲突。

术语一般如何实现？如何使用？

在 HuggingFace transformers 或自定义 RoPE 中设置 `rope_theta=1024`。p-RoPE 会略微降低短上下文 in-distribution 性能，但在长上下文泛化场景收益显著。LogN+p-RoPE 优于 LogN+RoPE，Scale-invariant p-RoPE 优于 Scale-invariant RoPE。

涉及论文标题：
- Scale-invariant Attention

---
