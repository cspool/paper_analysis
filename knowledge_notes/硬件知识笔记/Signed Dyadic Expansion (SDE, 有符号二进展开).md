## Signed Dyadic Expansion (SDE, 有符号二进展开)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Signed Dyadic Expansion (SDE) 是将实数近似为带符号的 2 的负幂次项求和的数学方法：SDE(r, K) = Σ_{k=1}^{K} a_k · 2^{-d_k}，其中 a_k ∈ {-1, +1}, d_k ∈ ℕ, d_1 < d_2 < ...。SDE 在硬件中的关键优势：乘以 2^{-d} 等价于右移 d 位，乘以带符号的 2^{-d} 等价于条件移位+取负，实现无需乘法器。LOGART 使用 SDE 近似 √2，构建贪心迭代算法：初始残差 r₁ = √2，每次迭代搜索 d_k 满足 2^{-d_k} ≤ |r_k| < 2^{-d_k+1}，a_k = sign(r_k)，r_{k+1} = r_k - a_k·2^{-d_k}。K-term 近似的绝对误差 ε_K = |r_{K+1}|。

从硬件架构角度拆解术语：
SDE 直接映射到 AE 的 Approx 模块（Figure 4(e)）：
```
# K=2 SDE for √2: √2 ≈ 1 + 0.5 = 2^0 + 2^{-1}
# Hardware: activation X 乘 √2
# X * √2 ≈ X * (1 + 0.5) = X + (X >> 1)
# 实现: 1 个移位器 + 1 个加法器，无乘法器

# K=3 SDE for √2: √2 ≈ 1 + 0.5 + 0.03125
# = 2^0 + 2^{-1} + 2^{-5}
# X * √2 ≈ X + (X >> 1) + (X >> 5)
# 实现: 2 个移位器 + 2 个加法器
```

术语一般如何实现？如何使用？
SDE 贪心算法：Python 中逐次逼近 `r = sqrt(2); terms = []; for k in range(K): d = ceil(-log2(abs(r))); terms.append((sign(r), d)); r -= sign(r)*2^{-d}`。硬件中 d_k 为整数移位量，a_k 选择加法或减法。K 越大精度越高但硬件开销越大（更多移位器和加法器）。LOGART 中 K=2 已足够保持精度。SDE 也适用于其他无理数（如 √3, log_2(3)）的硬件友好近似。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
