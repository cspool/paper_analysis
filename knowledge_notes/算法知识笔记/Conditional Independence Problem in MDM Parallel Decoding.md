## Conditional Independence Problem in MDM Parallel Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

条件独立性问题（Conditional Independence Problem）是掩码扩散模型（MDM）中并行解码面临的根本性质量问题。在MDM的τ-leaping推理中，多个[MASK]位置同时解码时，每个位置的token从其边际分布独立采样：p_j(X_{i_j}|E)。但真实的联合概率包含token间依赖：p(X_{i_1},...,X_{i_n}|E) = p(X_{i_1}|E)·p(X_{i_2}|X_{i_1},E)·...·p(X_{i_n}|X_{i_1},...,X_{i_{n-1}},E)。乘积边际分布忽略了条件依赖项，可能产生统计上不合理（但在各边际高概率）的token组合。Fast-dLLM论文给出了一个经典例子："The list of poker hands that consist of two English words are: _ _ " → 正确答案如"high card"或"two pair"，但独立采样可能产生"high house"——两个词分别高概率但组合不合理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

问题数学形式化：

```
# 假设解码位置i1和i2
# 边际分布独立采样:
x_i1 ~ p(X_i1 | x_context)        # p(X_i1="high")=0.4
x_i2 ~ p(X_i2 | x_context)        # p(X_i2="house")=0.3
# 乘积概率: q("high","house") = 0.4 × 0.3 = 0.12

# 真实联合分布:
# p(X_i1="full", X_i2="house") > 0.5   (full house常见)
# p(X_i1="high", X_i2="house") ≈ 0     (不存在的组合)
```

Fast-dLLM Theorem 1量化了条件独立假设与真实联合分布的偏差：

- 当每个token边际置信度 > 1-ε 且 (n+1)ε ≤ 1时，argmax的乘积分布 = argmax的真实分布（等价）
- 一般情况：L_p距离上界为 ((n-1)^p + 2n)^(1/p)·ε，TV距离 < (3n-1)ε/2
- 前向KL散度：D_KL(p||q) < (n-1)[H_b(ε) + ε·ln(|V|-1)]

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

缓解策略：(1) Fast-dLLM: 仅在高置信度token上并行解码（阈值τ≈0.7-0.9），低置信度token保持[MASK]留待后续步骤；(2) 降低每步并行token数（trade-off speed）；(3) 使用辅助模型显式建模token间依赖（如Discrete Copula Diffusion [Liu et al. 2024]）；(4) Block Diffusion: 通过块内自回归+块间扩散的半自回归方式，在块内保留token依赖。Fast-dLLM的factor策略使用理论绑定量(n+1)(1-c^(n)) < f动态选择安全并行token数，在速度和质量间取得最优平衡。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
