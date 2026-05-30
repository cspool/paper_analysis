## Affinity-Guided Quantization (AGQ / 亲和力引导量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Affinity-Guided Quantization (AGQ) 是 MoEQuant 论文提出的方法，通过在 PTQ 的量化误差计算中引入 token-expert 亲和力（即 gating coefficient c_i），解决 MoE 中不同 token 对同一 expert 贡献不均的问题。在 MoE 架构中，经过 gating network 的 softmax 后，每个 token 对其路由到的 expert 有一个权重 c_i，该权重链式传导到 expert FFN 的所有线性层（see Equation 17）。

传统 PTQ 假设所有 token 同等重要（量化损失为 L = Σ_i ||W x_i - W_hat x_i||_F^2），而 AGQ 重定义为 L = Σ_i c_i · ||W x_i - W_hat x_i||_F^2，使高亲和力 token 的量化误差惩罚更大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# AGQ: Affinity-Guided Quantization for MoE
# X: input activations to expert e [b, c]  (b tokens routed to this expert)
# c: gating coefficients for these tokens [b]  (from softmax after routing)
# W: weight matrix of expert e [o, c]

# ---- For AWQ-style methods (error-based) ----
# Original AWQ loss:
# L = ||WX - W_hat X||_F^2  = sum_i ||W x_i - W_hat x_i||^2

# AGQ-modified loss (Equation 18):
L_agq = 0
for i in range(b):
    error = ||W @ x_i - W_hat @ x_i||^2  # output error for token i
    L_agq += c[i] * error  # weight by gating coefficient

# ---- For GPTQ-style methods (Hessian-based) ----
# Original Hessian: H = X @ X^T
# Shape: [c, c]

# AGQ-modified Hessian (Equation 19):
sqrt_c = sqrt(c)  # [b]
X_weighted = X * sqrt_c.reshape(-1, 1)  # [b, c], broadcast
H_agq = X_weighted.T @ X_weighted  # [c, c]
# Equivalent to: H_agq = X.T @ diag(c) @ X
```

物理含义：c_i 体现了 token i 与 expert e 的"相关度"。对 router 高度信任的 token（c_i 大），AGQ 赋予更大的量化误差权重，确保这些关键 token 的输出质量得到更好保护。反之，对 c_i 极小的 token，量化误差的影响也较小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AGQ 在 MoEQuant 框架中作为插件式模块实现，与 GPTQ 或 AWQ 无缝集成。对于 AWQ，AGQ 修改量化损失函数中的 per-token 权重；对于 GPTQ，AGQ 修改 Hessian 矩阵计算。论文实验表明 AGQ 单独使用带来约 2% 的平均分提升（DeepSeek-MoE-16B 上），而 EBSS + AGQ 结合使用提升约 2.6%。在 Mixtral-8x7B 上 AGQ 单独效果不如 baseline GPTQ，但 EBSS 和 AGQ 联合使用仍是最优配置，说明了两种方法的互补性。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
