## Frontier Search (Monotonicity-Based Threshold Optimization)

术语是什么？
Frontier Search是MoDES提出的基于单调性的二维阈值搜索算法。在B×B空间中（100个grid points × 100个grid points）找到满足target skipping ratio ρ约束下最小化KL divergence f的最优(τ_t, τ_v)对。利用f和g（skipping ratio）关于参数的单调非递减性，将O(ND²) exhaustive search降至O(ND)，搜索时间~45×降低（>2天→<2小时），最优解性能差异<0.01%。

从算法pipeline角度拆解术语：
```
FrontierSearch(B={τ^{(1)},...,τ^{(D)}}, ρ):
    frontier = ∅; p = D
    for q = 1 to D:                      // increasing τ_t
        while p ≥ 1 and g(τ^{(q)}, τ^{(p)}) ≥ ρ:
            p = p - 1                    // monotonicity: shrink τ_v
        p_{(q)} = p + 1
        if p_{(q)} ≤ D:
            compute f(τ^{(q)}, τ^{(p_{(q)})})
            frontier ∪= {(q, p_{(q)})}
    return argmin_{(q,p)∈frontier} f     // optimal (τ_t*, τ_v*)
```
关键性质：p_{(q)}关于q非递增（更大的τ_t→需更小τ_v满足ρ）；最优解必在frontier上（非frontier解被dominated）。

术语一般如何实现？如何使用？
在calibration set C（1024 GQA samples）上evaluate，每对(τ_t, τ_v)需1次forward pass。Frontier search总forward pass≤200（vs naive 10,000）。对20-30B MLLM，calibration+search在8×H200上20分钟至<4小时。D=100经ablation验证足够。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---
