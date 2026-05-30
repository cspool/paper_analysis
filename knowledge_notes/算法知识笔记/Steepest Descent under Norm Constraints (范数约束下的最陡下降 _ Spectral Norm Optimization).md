## Steepest Descent under Norm Constraints (范数约束下的最陡下降 / Spectral Norm Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Steepest Descent under Norm Constraints 是 Bernstein & Newhouse (2024) 提出的优化理论框架。核心观点：深度学习的每一步优化可视为在某种范数约束下寻找使损失下降最快的方向（最陡下降方向）。形式化地：ΔW = argmin_{||Δ|| ≤ η} ⟨∇L, Δ⟩，其中范数 ||·|| 的选择决定了优化器的行为。在此框架下：
- Adam/AdamW 可解释为 Max-of-Max norm 约束下的最陡下降（动态调整的逐元素范数约束）
- Muon 可解释为 spectral norm（或大 p 的 Schatten-p norm）约束下的最陡下降——当 Newton-Schulz 精确计算时，Muon 的谱范数约束意味着更新矩阵的奇异值被限制为 1，即更新在所有方向上等强度
- 从数学角度看，权重矩阵作为输入/隐空间上的 operator，其自然范数应为 induced operator norm（spectral norm），因此 Muon 的 norm constraint 比 AdamW 的逐元素约束更合理

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
该理论对 Muon 的设计提供了数学解释：

```
# 一般的最陡下降框架：
给定 norm ||·||，每一步求解：
ΔW* = argmin_{||Δ|| ≤ η} ⟨∇L(W), Δ⟩

# 不同的 norm 选择对应不同的优化器：
# - ||Δ||_∞（element-wise max norm）→ 符号梯度下降 (signSGD)
# - Max-of-Max norm（动态自适应）→ Adam/AdamW
# - ||Δ||_2（spectral norm）→ Muon (当 Newton-Schulz 精确时)
# - ||Δ||_{S_p}（Schatten-p norm, p 大）→ Muon 近似实现

# Muon 如何实现 spectral norm 约束下的最陡下降：
M = momentum(∇L)                           # 先累积动量
O = Newton-Schulz(M)                        # O ≈ U V^T
                                            # O 的奇异值 = 1 (精确时) 或 ≈ 1 (近似)
                                            # ||O||_2 = 1, ||O||_F = √r
ΔW = -η * O                                # 谱范数 = η (受约束)
```

该视角还揭示了 Muon 与 Shampoo 的关系：当去掉 Shampoo 中的 preconditioner accumulation 后，Shampoo 的更新退化为 W_{t+1} = W_t - η U V^T（即无动量的 Muon = spectral descent）。移除 preconditioner 等价于将优化问题拉伸为各向同性——这正是矩阵正交化所做的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 该理论框架本身不是可执行的算法，而是理解已有优化器行为的概念工具
- 实践含义：(a) 选择优化器 = 选择 norm constraint 类型，应根据参数结构选择（矩阵参数 → spectral norm，向量参数 → Euclidean norm）；(b) Muon 的设计由此推广到 Schatten norm——未来工作方向是在 Muon 框架中引入 Schatten-p norm 支持（论文 Sec 4 讨论），可能通过调整 Newton-Schulz 多项式实现不同的奇异值变换；(c) 该框架解释了为什么 Muon 与 AdamW 结合使用是合理的——非矩阵参数（norm、bias、embedding）的适当范数是 Euclidean/逐元素范数（AdamW），而矩阵参数的适当范数是 spectral norm（Muon）
- 相关代码：Bernstein & Newhouse 的分析在 [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)；Cesista (2024) 的博客提供了可视化解释

涉及论文标题：
- Muon is Scalable for LLM Training
