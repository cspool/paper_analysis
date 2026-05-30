## Energy Function for Self-Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Energy Function for Self-Attention 是 Tree Attention 论文推导出的将 self-attention 操作表达为标量能量函数梯度的数学公式。能量函数定义为：
$$F(\zeta) = \log \sum_{a=1}^{N} \exp(q \cdot k_a^T + \zeta \cdot v_a^T)$$
其中 $\zeta \in \mathbb{R}^{d_h}$ 是辅助 "source" 向量，q,k,v 是 query/key/value。self-attention 操作是 F 关于 ζ 在 ζ=0 处的梯度：
$$\sum_{a=1}^{N} \operatorname{softmax}(q \cdot k_a) v_a = \left. \frac{\partial F}{\partial \zeta} \right|_{\zeta=0}$$

这个公式的物理意义来自统计力学：F 是 cumulant-generating function（类比 Helmholtz 自由能），softmax 后的 attention scores 构成概率分布 P_a = exp(q·k_a^T) / Σ_i exp(q·k_i^T)（partition function Z = Σ exp(q·k^T)），引入 source ζ 后 Z(ζ) = Σ exp(q·k_a^T + ζ·v_a^T)，attention 输出 = ⟨v⟩ = (1/Z) ∂Z/∂ζ|_{ζ=0} = ∂logZ/∂ζ|_{ζ=0}。

从算法pipeline角度拆解术语。
能量函数在 Tree Attention pipeline 中的作用：
```
给定: q, K_i (本地 key chunk), V_i (本地 value chunk)

# 能量函数 (forward, Algorithm 1):
r_i = q·K_i^T + ζ·V_i^T          # 每个 chunk 的 "能量贡献", [t]
m = TreeReduce(max, r_i)         # 全局 max (numerical stability)
r_i' = r_i - m                    # stable shift
F = TreeReduce(logsumexp, r_i')  # 全局 logsumexp = 能量函数值

# 梯度 (backward w.r.t ζ, Algorithm 2):
∂F/∂ζ = TreeReduce(sum, exp(r_i' - F) · V_i)  = attention 输出
```

关键洞察：自动微分的经典结论——∇f(x) 可以用与 f(x) 相同的渐进时间复杂度计算。因此，如果能高效计算能量函数 F（通过 tree reduction O(N/p + log p)），就能高效计算 attention（F 的梯度）。F 的 computational graph 很浅（仅 3 次 AllReduce），因此反向传播的内存开销可忽略。

术语一般如何实现？如何使用？
实现：在代码中，Algorithm 1 和 Algorithm 2 被合并为单一函数 `tree_flash_decode`（Appendix D），同时返回能量函数值和梯度（即 attention 输出）。ζ 实际上不会被 materialize——ζ=0 时，F 退化为仅关于 q·k 的 logsumexp，梯度计算简化为 exp(q·k - lse)·v 的加权和。

用途：能量函数表述不仅是理论好奇心——它直接揭示了 attention 计算中的结合律结构（logsumexp 的 associative property），从而证明了 tree reduction 的可行性，为 Tree Attention 算法提供了数学正确性保证。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters
