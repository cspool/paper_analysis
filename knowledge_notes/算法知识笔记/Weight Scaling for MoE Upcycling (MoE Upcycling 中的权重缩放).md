## Weight Scaling for MoE Upcycling (MoE Upcycling 中的权重缩放)

术语是什么？

Weight Scaling 是 NVIDIA 为 MoE upcycling 提出的权重初始化补偿技术。在 upcycling 过程中，expert 输出因 Router softmax 和 fine-grained sharding 的综合效应被缩小——对 E×G 个 expert、top-T 路由，初始每个 expert 的输出大约被缩放为原来的 1/(E×G)，而 T 个 expert 的加权和大致等于 (T/(E×G²)) × dense_output。为了补偿这个缩放，对每个 expert 的 W1 和 W2 权重同时乘以缩放因子：

$$\text{scale} = \sqrt[3]{\frac{E \times G^2}{T}}$$

该公式对 Squared-ReLU 激活函数从第一性原理推导得出，但实验证明对 SwiGLU 激活同样有效。Weight Scaling 同时适用于 coarse-grained MoE（G=1 时 scale = ³√(E/T)）和 fine-grained MoE。

从算法pipeline角度拆解：

Weight Scaling 的推导（以 Squared-ReLU 为例）：

```
# MoE 激活（uniform distribution 假设, iteration 0）:
# P = P_1 = P_2 = ... = P_T = 1/(E*G)
MoE_activation = P * sum_{i=1}^{T} E_i(x)
               = (1/(E*G)) * (T/G) * dense_activation
               = T/(E*G^2) * dense_activation

# Squared-ReLU: output = W2 @ (ReLU(W1 @ x))^2
# 性质: squared_relu(k*w) = k^2 * squared_relu(w)
# 所以: 若 W1 *= k1, W2 *= k2
#       expert_output *= k1^2 * k2

# 需要 k1^2 * k2 = E*G^2/T  (补偿缩放)
# 取 symmetrically: k1 = k2 = (E*G^2/T)^{1/3}
```

实验验证：
- 对 E8G8T8 (64 experts top-8, 1/8 size): scale = ³√(8×64/8) = ³√64 = 4.0
- 对 E8G1T1 (8 experts top-1): scale = ³√(8×1/1) = 2.0
- w/ weight scaling 比 w/o 低 1.5% loss (Nemotron-4 15B E8G1T1)

术语一般如何实现？

在 Megatron-LM upcycling 初始化代码中，计算缩放因子后对每个 expert 的 W1 和 W2 进行 element-wise 乘法。论文同时尝试了替代方案（MoE output scaling 和 post expert layernorm），但 weight scaling 效果最优且实现最简单（不改变模型架构）。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
