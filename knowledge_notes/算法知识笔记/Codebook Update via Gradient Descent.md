## Codebook Update via Gradient Descent

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Codebook Update via Gradient Descent 是 GPTVQ 在完成 Algorithm 1 的逐 block 量化后，对 codebook 值进行梯度下降微调的优化步骤。GPTVQ 的逐 block 量化（Algorithm 1+2）已经确定了每个权重的质心索引 I，但质心值 C 是贪心初始化的。通过固定索引 I，以层输出 MSE 为目标进一步优化质心值 C：min_C ||WX - Q(C)X||²_F，其中 Q(C) 是基于 C 和固定索引 I 的查找重建操作（look-up operation）。该目标对 C 是凸二次规划问题，可用闭式解但由于大矩阵求逆开销高，GPTVQ 使用 PyTorch 梯度下降代替——每步更新 C 后重建 Q，梯度 ∂Q/∂C 因查找操作而简单定义。消融（Table 15）显示 codebook update 在所有 setting 下降低最终 perplexity，代价是中等额外运行时间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GPTVQ Codebook Update 伪代码
# 前提: Algorithm 1 和 2 已完成，索引 I 已确定，Q(W) 已构造
# 输入: 原始权重 W，校准输入 X，当前质心 C，索引 I
# 输出: 优化后的质心 C_optimized

# 构建基于 C 和 I 的查找重建函数
def reconstruct_Q(C, I):
    # I[r, c] = 质心索引 (0..k-1)
    # Q[r, c] = C[I[r, c]]  # 查找操作
    return Q

# 用 Adam 梯度下降优化 C
C_param = nn.Parameter(C.clone())
optimizer = Adam([C_param], lr=1e-3)

for step in range(N_steps):  # 通常 ~1000 steps
    Q_hat = reconstruct_Q(C_param, I)  # 用当前 C 重建 Q
    loss = ||W @ X - Q_hat @ X||²_F  # 层输出 MSE
    loss.backward()  # ∂loss/∂C 通过查找操作传播
    optimizer.step()
    optimizer.zero_grad()

C_optimized = C_param.detach()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 使用 PyTorch 的 autograd 实现梯度下降（比闭式解伪逆更快）。每个 group 的 codebook 独立优化，group 间无交互。梯度 ∂Q/∂C 的实现：因 Q[r,c] = C[I[r,c]]（索引查找），∂loss/∂C[m] = Σ_{(r,c): I[r,c]=m} ∂loss/∂Q[r,c]。实现要点：(1) N_steps 通常为数百到一千（消融显示 1000 steps 足够收敛）；(2) 可选对 codebook 值施加 L2 正则化防止过拟合；(3) Codebook update 在 quantization 后、模型导出前执行。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---
