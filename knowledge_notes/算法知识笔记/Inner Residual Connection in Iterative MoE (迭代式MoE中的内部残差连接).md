## Inner Residual Connection in Iterative MoE (迭代式MoE中的内部残差连接)

术语解释
Inner Residual Connection 是 CoE 中用于稳定多步 expert 迭代训练的关键设计：在每一步 expert 输出后立即加入残差连接 $x^{(t)} = \text{expert\_out} + x^{(t-1)}$，而非只在最后一步之后加一次 outer residual（$y = x^{(C)} + x^{(0)}$）或每步都加初始输入（init residual: $x^{(t)} = ... + x^{(0)}$）。

术语是什么？
三种 residual 设计对比（CoE 论文）：
- **Inner Residual（默认，loss=1.12）**：$x^{(t)} = \text{expert\_out}^{(t)} + x^{(t-1)}$，每一步加入前一步输出
- **Outer Residual（loss=1.21）**：$y = x^{(C)} + x^{(0)}$，仅在最终输出加残差
- **Init Residual（loss=1.18）**：$x^{(t)} = \text{expert\_out}^{(t)} + x^{(0)}$，每步都加原始输入

从算法pipeline角度拆解术语：
```
# Inner Residual（CoE默认，最佳）
for t in 1..C:
    out = sum(g[t,i] * E_i(x_cur))
    x_cur = out + x_cur              # 残差来自上一步

# Outer Residual（消融，差）
for t in 1..C:
    out = sum(g[t,i] * E_i(x_cur))
    x_cur = out                      # 无中间残差
y = x_cur + x_0                      # 仅最后加一次

# Init Residual（消融，居中）
for t in 1..C:
    out = sum(g[t,i] * E_i(x_cur))
    x_cur = out + x_0                # 残差始终来自原始输入
```

术语一般如何实现？如何使用？
- Element-wise addition，计算开销可忽略
- 作用：为梯度提供从 $x^{(C)}$ 到 $x^{(t)}$ 的直接路径，稳定 credit assignment
- 与标准 Transformer residual connection 的区别：不是跨层（layer-to-layer），而是跨 iteration（iteration-to-iteration within single layer）

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

---
