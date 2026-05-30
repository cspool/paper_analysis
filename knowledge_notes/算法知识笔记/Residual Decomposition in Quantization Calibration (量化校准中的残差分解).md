## Residual Decomposition in Quantization Calibration (量化校准中的残差分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
残差分解（Residual Decomposition）是 GPTAQ 中解决非对称校准效率瓶颈的关键技术。在非对称校准中，每次量化一列权重后需要重新评估输出残差 R = W X̃ − W X。直接计算复杂度为 O(mnk)，其中 k（token 数 × 校准样本数）远大于 m 和 n（例如 LLaMA2-7B 中 k = 128×2048 ≈ 262K，而 n = 4096），重复计算 R 将导致不可接受的量化时间。GPTAQ 的关键洞察：输出残差 R 可分解为 n 个独立神经元分量的和——R = W ΔX = Σ_{q=1}^n W_{:,q} ΔX_{q,:}。这样：(1) 可一次性计算 ΔX = X̃ − X（O(nk)）；(2) 第 q 次迭代仅关注第 q 个神经元对残差的贡献 `W_{:,q} ΔX_{q,:}`，而非全部 n 个分量；(3) 优化目标简化为 `min ||ΔW_{:,q:} X_{q:,:} − W_{:,q} ΔX_{q:,:}||²`，最优权重更新包含 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}` 项；(4) 由于 `ΔX_{q,:}X_{:,q:}^T H_{-q:}^{-1}` 与权重更新无关，可预计算为 P 矩阵的对应行。这一分解将复杂度从 O(mn²k) 降至 O(mn² + n²k)（降低 n 倍），使得 GPTAQ 的非对称校准在实际中可用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
残差分解的数学转化：

```
# 直接法（不可行）：
for q in range(n):                  # n = 4096 (hidden_dim)
    quantize_and_update(W[:, q])
    R = W @ (X̃ - X)                 # O(mnk), k=262K → 每次迭代 ~43GB 运算
    # 对全矩阵 R 做 Hessian 逆投影
    correction = R @ X.T @ H_inv    # O(mn² + n²k), 又一大运算
# 总复杂度: O(n × mnk) = O(mn²k)，完全不实际

# 残差分解法（GPTAQ）：
# 预计算（一次性）:
ΔX = X̃ - X                         # O(nk)
ΔX_XT = ΔX @ X.T                   # O(n²k)

# 将 R 写为分解形式:
# R = W @ ΔX = Σ_{q=1}^n W[:,q] × ΔX[q,:]
#   = sum of n rank-1 matrices, each contributed by one neuron

# 第 q 次迭代:
# 仅关注第 q 个神经元残差分量: W[:,q] × ΔX[q,:]
# 优化目标: min ||ΔW[:,q:] @ X[q:,:] − W[:,q] @ ΔX[q:,:]||²
# 权重更新 (Eq. 15):
ΔW[:,q:] = GPTQ_term + W[:,q] @ (ΔX[q,:] @ X[q:,:].T @ H_{q:}^{-1})
#                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                           P[q,q:] 的第 q 行, 可预计算!

# P 矩阵通过 Theorem 4.2 一次性并行计算:
P = ((ΔX_XT @ L) ⊙ M_U) @ L.T     # GPU 上 <1ms (Fig.4a)
```

**Annotations**: 关键洞察是 R 可以按神经元维度分解——每个输出通道（W 的每一行）对 R 的贡献来自输入通道（W 的每一列）× 该输入通道的激活偏差 ΔX[q,:]。因此第 q 次迭代只处理与第 q 个输入神经元相关的分量，而第 1..q-1 个神经元的权重已被固定（量化完成），相应的残差分量已被"吸收"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
残差分解在 GPTAQ 中的实现与 Cholesky 重构化协同：(1) ΔX X^T 矩阵在校准循环外一次性计算（n×n 矩阵，对 hidden_dim=4096 的层约 64MB FP16）；(2) 利用 Cholesky 分解的 L 矩阵和 Theorem 4.2 将 P 矩阵计算并行化为一句话：`P = ((ΔX_XT @ L) * mask_upper_tri) @ L.T`；(3) 在 lazy-batch 更新中直接切片 P[Q,Q:] 与 W[:,Q] 做 rank-1 更新。残差分解的前提是激活量化在权重量化之前执行（A→W 顺序），使得 ΔX 不仅是权重偏差还包含激活量化偏差。GPTAQ 的内存分析：ΔX 临时需要 O(nk) 存储（LLaMA2-7B 约 12GB），但逐 block 处理后可释放；P 矩阵 O(n²)（每层 0.16-0.70GB）需保留在 GPU 内存中供 lazy-batch 迭代使用。残差分解和 Cholesky 重构化、lazy-batch 更新的组合使得 GPTAQ 的额外延迟控制在大维度时 30-40%、小维度时 <10%，而非比 GPTQ 慢 n 倍。

涉及论文标题：
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration



---
