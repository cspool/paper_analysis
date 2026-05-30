## Pyramid-Shaped KV Cache Allocation (金字塔形 KV Cache 分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pyramid-Shaped KV Cache Allocation 是一种层间（跨层）KV Cache 预算分配策略。核心思想：浅层 Transformer 层的注意力分布更均匀（不稀疏），深层注意力则高度集中在少数 token 上（注意力稀疏性）。因此，在不同层之间统一保留相同数量 token 的 KV Cache 是次优的——应为浅层分配更多 KV Cache 预算、为深层分配更少，形成"金字塔形"分配。PyramidInfer (Yang et al., 2024) 和 PyramidKV (Cai et al., 2024) 率先提出此策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**金字塔形 KV Cache 分配的数学定义**：

全局参数：$r$ 为总 KV cache 保留率，$l$ 为序列总长度，$l_w$ 为观察窗口长度，$l_c = l - l_w$ 为上下文中 eviction 候选 token 数。

上下文的保留率：
$$r_c = \frac{r \cdot l - l_w}{l_c}$$

设 $\beta = 0.05$（最小保留率），$\alpha = \frac{1}{2}(1+\beta) = 0.525$。层 0 和层 $m-1$ 的保留率：
$$r_c(0) = \begin{cases} 2 \times r_c - 0.05, & \beta < r_c \le \alpha \\ 1, & \alpha < r_c \le 1 \end{cases}$$

$$r_c(m-1) = \begin{cases} 0.05, & \beta < r_c \le \alpha \\ 1 - 2 \times r_c, & \alpha < r_c \le 1 \end{cases}$$

第 $\lambda$ 层保留率（线性插值）：
$$r_c(\lambda) = r_c(0) + \frac{r_c(m-1) - r_c(0)}{m-1} \cdot \lambda$$

**具体例子**（$m=32$, $l=4096$, $l_w=128$, $r=0.40$）：
```
r_c = (0.40 × 4096 - 128) / (4096 - 128) = 0.381
r_c(0) = 2 × 0.381 - 0.05 = 0.712      # Layer 0 保留 71.2%
r_c(31) = 0.05                            # Layer 31 保留 5.0%
r_c(15) = 0.712 + (0.05-0.712)/31 × 15 = 0.392
```

推理时每层执行：
```
ac_i = accumulated_attention_scores(Q, K, l_w)
k = floor(r_c(λ) × l_c)
η_i = argTopK(ac_i, k)
Γ_{r,i} = Γ_i[η_i]
```

术语一般如何实现？如何使用？

PyramidKV 开源见 https://github.com/Linking-ai/PyramidKV，PyramidInfer 见 https://github.com/mutonix/pyramidinfer。主要区别：PyramidInfer 的 eviction 是逐层的——layer i 被 evict 的 token 在 layer i+1 也不重新计算；PyramidKV 的 KV cache 在不同层独立管理。SpindleKV 沿用了 PyramidKV 方式，并在其基础上增加浅层码本压缩。

涉及论文标题：
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

---
