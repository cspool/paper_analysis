## Averaging Approximation for PTQ Hessian（PTQ Hessian 的平均近似）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Averaging Approximation 是 GuidedQuant 为解决 block-diagonal Fisher 计算不可行问题而提出的近似技术。直接为每层存储 d_out 个 d_in×d_in 的 Fisher block 矩阵 `H_j = XᵀDiag((∂ℓ/∂z_j)²)X` 需要 Θ(d_in² d_out) 存储——Llama-2-7B 的 self-attention projection 层（d_in=4096~11008, d_out=4096~11008）需远超可用 GPU 内存。Averaging approximation 将 output channels 划分为 g 个组（g≪d_out），每组内平均所有 block 的 Fisher 信息：`H̄_k = XᵀDiag( (1/|J_k|) Σ_{j∈J_k} (∂ℓ/∂z_j)² ) X`。存储从 Θ(d_in² d_out) 降至 Θ(d_in² g)，通过连续的 output channels 成组实现。实验表明（Table 13）g=1 已捕获大部分性能增益（2-bit: Wiki2 9.00 vs g=4 的 8.83），g=2 几乎达到饱和。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 直接法（不可行）：d_out 个 d_in×d_in 矩阵
for j in 1..d_out:
    s_j = (∂ℓ/∂z_j)²                           # n 维向量
    H_j = Xᵀ @ Diag(s_j) @ X                   # d_in × d_in
# 存储: d_out × d_in²，Llama-2-7B 一层需 >200GB

# Averaging approximation（可行）：g 个 d_in×d_in 矩阵
for k in 1..g:
    J_k = {d_out*(k-1)/g + 1, ..., d_out*k/g}  # 连续 channel 分组
    s̄_k = (1/|J_k|) * Σ_{j∈J_k} (∂ℓ/∂z_j)²    # 平均梯度平方
    H̄_k = Xᵀ @ Diag(s̄_k) @ X                   # 共享 Hessian
# 存储: g × d_in²，g=4 时一层 ~0.4GB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GuidedQuant 使用连续分组策略（默认每 d_out/g 个连续 channels 一组），实验显示此简单策略有效，可能 benefit from more sophisticated clustering。超参数 g 的选择权衡：g 越大 → Fisher 近似越精确（类似 block-diagonal → 接近 full Fisher within groups）→ 性能越好（边际递减）；g 越大 → 存储/计算开销越大（线性增长）。推荐值：g=4 for 7B/13B, g=2 for 70B。g=1（全局平均，退化为 SqueezeLLM 风格的对角近似去掉所有跨权重依赖）在极端压缩下仍有显著提升。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
