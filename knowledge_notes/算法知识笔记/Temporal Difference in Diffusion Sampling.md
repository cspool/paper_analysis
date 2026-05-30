## Temporal Difference in Diffusion Sampling

术语是什么？
Temporal Difference（时序差分）在扩散模型上下文中指相邻扩散时间步（t 和 t+1）之间激活的差值 $\mathbf{a}_t^{(l)} - \mathbf{a}_{t+1}^{(l)}$。MoDiff论文首次系统研究了扩散采样过程中时序差分的统计性质——其分布范围比原始激活小10×以上（Figure 1b橙色vs蓝色violin plot高度对比），分布更集中在零附近，且在不同时间步之间范围更一致，几乎无异常值。这一统计性质是MoDiff调制量化有效性的理论基石。

时序差分之所以具有这些有利属性，是因为扩散采样过程的连续性质：相邻时间步的去噪输出高度相似（这也是DeepCache等方法能够复用缓存的基础），因此它们的差分自然地落在更小的数值范围内。

从算法pipeline角度拆解术语：
```
// 时序差分的统计性质（基于CIFAR-10 DDIM, 100 steps）

// 原始激活分布：
a_t^{(l)} ~ LongTail(μ_t, σ_t²)     // 不同t的μ_t、σ_t变化大
range(a) ≈ [−R, R]                   // R ≈ O(10)
outliers: P(|a| > 3σ) 显著           // 长尾分布导致scaling factor被极端值主导

// 时序差分分布：
d_t^{(l)} = a_t^{(l)} - a_{t+1}^{(l)}
d_t^{(l)} ~ Concentrated(≈0, σ_d²)   // 集中在0附近
range(d) ≈ [−r, r], r < R/10         // 范围小10×以上
outliers: P(|d| > 3σ_d) 几乎为0      // 无长尾

// Theorem 4.3量化误差分析：
Err_Q(a) = (2R)² × d / (2^b - 1)²
Err_Q(d) = (2r)² × d / (2^b - 1)²
Err_Q(d) / Err_Q(a) = (r/R)² < 1/100  // 100×+ reduction
// 等效：用相同误差界可将位宽降低 log₂(R/r) ≈ 3-4 bits
```

术语一般如何实现？如何使用？
时序差分的统计性质不需要额外计算——它是扩散采样过程的固有属性。MoDiff利用它：将量化对象从原始激活切换为时序差分（无额外数据依赖或预处理）。使用时仅需在每层线性算子的forward中做一次矩阵减法（a_t - â_{t+1}）。论文在多个数据集/模型上验证了时序差分性质的普适性（CIFAR-10 DDIM、LSUN LDM、Stable Diffusion、DiT），证明不限于特定架构或采样器。

涉及论文标题：
- Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

---
