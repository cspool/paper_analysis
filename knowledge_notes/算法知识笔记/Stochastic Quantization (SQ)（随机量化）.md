## Stochastic Quantization (SQ)（随机量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stochastic Quantization (SQ) 是一种无偏随机舍入方法。给定实数值 x ∈ [a, b] 和两个量化值 a, b，SQ 以概率 p_a = (b-x)/(b-a) 输出 a，以概率 p_b = (x-a)/(b-a) 输出 b，使得 E[x̂] = x（无偏性）。方差 Var[x̂] = (b-x)(x-a)，在区间中点处最大（= (b-a)²/4），在端点处为零。与 Round-to-Nearest (RTN) 不同（确定性、有偏），SQ 的随机性保证了无偏性，这对分布式场景（多客户端量化后聚合）至关重要：当 n 个独立无偏估计量平均时，MSE 随 1/n 衰减；而有偏量化中误差可能不随 n 衰减（因偏差相关）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SQ 在单个值的随机量化流程：

```
输入: x ∈ R, 量化值 a ≤ x ≤ b

// 计算概率
p_a = (b - x) / (b - a)
p_b = (x - a) / (b - a)

// 随机舍入
sample u ~ Uniform(0, 1)
if u < p_a:
    x̂ = a
else:
    x̂ = b

// 性质验证
E[x̂] = a·p_a + b·p_b
      = a·(b-x)/(b-a) + b·(x-a)/(b-a)
      = (ab - ax + bx - ab) / (b-a)
      = x·(b-a)/(b-a) = x  ✓ 无偏

Var[x̂] = (x-a)²·p_a + (x-b)²·p_b
        = (x-a)²·(b-x)/(b-a) + (b-x)²·(x-a)/(b-a)
        = (b-x)(x-a)  ✓
```

向量级量化：对 X = (x_1,...,x_d)，使用量化值集合 Q = {q_1,...,q_s}，每个 x_i 被包围它的连续量化值对 [q_j, q_{j+1}] 独立随机量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SQ 是众多分布式学习通信压缩方案的基础构建块。QSGD [Alistarh et al., NeurIPS 2017] 使用全局范数确定量化值（非自适应），NUQSGD [Ramezani-Kebrya et al., JMLR 2021] 使用全局 min/max。ASQ 与这些方法的区别在于：ASQ 使用优化的 Q 集合对 SQ 做"自适应"增强——SQ 本身是"如何量化单个值"的机制，ASQ 是"如何选择最优 Q 集合"的问题。SQ 实现简单（仅需一随机数生成器），可与 sparsification、top-k、Huffman coding 等正交压缩技术叠加。

涉及论文标题：
- Optimal and Approximate Adaptive Stochastic Quantization
