## Iteration-based Independent Routing (基于迭代的独立路由)

术语解释
Iteration-based Independent Routing 是 CoE 架构的核心路由机制：在 C 步 expert 迭代处理的每一步，使用独立的 Router 参数 $W_{router}[t] \in \mathbb{R}^{d \times N}$（而非跨步共享同一 Router）。共享路由的消融变体（所有步骤复用同一 Router 和 gating）validation loss plateau 在 ~1.5，远差于独立路由的 1.12。

术语是什么？
形式化：第 t 步 $g_{t,i} = \text{TopK}(\text{Softmax}(e_{t,i}^\top x^{(t-1)}), K/C)$，每步 Router 参数独立。与 token-level 动态路由（Ada-K, DynMoE）的区别：后者是同一层内不同 token 使用不同 k 值但 Router 参数固定；CoE 是同一 token 在不同 iteration 使用不同 Router 参数。

从算法pipeline角度拆解术语：
```
# 独立Router（CoE默认，性能好）
for t in 1..C:
    logits = x_cur @ W_router[t]    # 每步不同W_router

# 共享Router（消融变体，性能差~1.5 loss）
logits = x @ W_router               # 仅第一步计算
topk_idx = TopK(Softmax(logits), K)
for t in 1..C:
    expert_out = sum(g[i] * experts[i](x_cur) for i in topk_idx)
    x_cur = expert_out + x_cur      # gating固定不变
```

术语一般如何实现？如何使用？
- 参数量增加 C 倍 Router 参数（通常可忽略，C=2 且 Router 参数占总参数比例极低）
- Co-activation 矩阵验证：不同 iteration 的 expert 选择集合高度非对称，证明路由决策确实随 iteration 变化
- 适用于需要 multi-step reasoning 的任务
- 是 CoE 中两个不可消融的组件之一（与 inner residual 并列）

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

---
