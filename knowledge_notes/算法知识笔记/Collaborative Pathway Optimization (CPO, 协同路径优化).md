## Collaborative Pathway Optimization (CPO, 协同路径优化)

术语解释
Collaborative Pathway Optimization 是 C3PO 的核心优化范式：利用参考集中多个"成功样本"的 expert pathway 来协同优化测试样本的 pathway。与传统的 prompt tuning（优化连续 prompt embeddings）或 ICL（拼接示例到输入）不同，CPO 直接操作低维的 routing weights 空间，且利用邻居样本之间的协作（collaboration）而非孤立优化。

术语是什么？
CPO 的"协同"体现在三个层面：
1. **邻居协同**: 不是用单个最相似样本的 pathway，而是用 k 个邻居的 pathway 加权融合（kernel weighting）
2. **跨层协同**: pathway 矩阵同时编码了所有层的 routing weights，优化时跨层信息自然交互
3. **参考集协同**: 参考集中的样本互不重叠，但通过 kernel 函数为不同测试样本提供不同的协同信号

CPO 的三种具体形式：
- **Mode Finding (Meanshift)**: 梯度自由，在 ω-space 中找到邻居 pathway 的最密集模式
- **Kernel Regression**: 梯度自由，用邻居 pathway 的核加权平均作为目标
- **NGD**: 梯度方法，用邻居 loss 的加权平均作为 surrogate objective

从算法pipeline角度拆解术语：
CPO 的通用框架伪代码：

```
def CPO(test_sample x, model f, reference_set D_ref, method="NGD"):
    # Step 1: 嵌入检索
    emb = embed(x)
    neighbors = knn(emb, D_ref.embeddings, k=3)
    
    # Step 2: 提取当前 pathway
    ω_curr = extract_routing_weights(f, x, layers=last_5, experts=top_20)
    
    # Step 3: 协同优化
    if method == "ModeFinding":
        ω_new = meanshift(ω_curr, neighbors.omegas, kernel="gaussian")
    elif method == "KernelRegression":
        ω_hat = sum(K(x, xi) * ω_i for xi, ω_i in neighbors) / sum(K(x, xi))
        α = argmin_α surrogate_loss(α*ω_curr + (1-α)*ω_hat)
        ω_new = α*ω_curr + (1-α)*ω_hat
    elif method == "NGD":
        for step in range(10):
            loss = weighted_average([loss(f(xi, ω), yi) for xi, yi in neighbors])
            ω_curr -= lr * ∇_ω loss
    
    return f.forward(x, routing_override=ω_new)
```

CPO vs 其他 test-time adaptation 方法：

| 方法 | 优化变量 | 变量维度 | 需要反向传播 | 参考集需求 |
|------|---------|---------|------------|----------|
| ICL | 无（拼接示例） | 0 | 否 | 大（few-shot examples） |
| Prompt Tuning | Soft prompt tokens | d×len(prompt) | 是 | 全量参考集 |
| Prefix Tuning | Prefix vectors | L×d×len(prefix) | 是 | 全量参考集 |
| CPO (Kernel Reg) | Routing weights | L×E (subset) | 否 | 仅 kNN 邻居 |
| CPO (NGD) | Routing weights | L×E (subset) | 是 | 仅 kNN 邻居 |

术语一般如何实现？如何使用？
- 实现与 Test-Time Expert Re-Mixing 共享代码框架
- 关键超参数：k=3（kNN 邻居数），steps=10（NGD 优化步数），Gaussian kernel（核函数选择），cosine annealing LR（学习率调度）
- NGD 收敛快：前 6 步贡献 +11.6% accuracy gain，10 步后 plateau
- 仅 5.1% 的初始正确预测在优化后被翻转为错误（稳定性好）

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
