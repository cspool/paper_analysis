## Frontier Search for Threshold Optimization

术语是什么？
Frontier Search 是 MoDES 提出的一种利用单调性属性加速阈值对搜索的优化算法。在 DMT 中需要确定最优的 $(\tau_t, \tau_v)$ 对，使 skipping ratio 满足目标约束 $\rho$ 的同时，最小化与原始模型输出的 KL 散度。Naive exhaustive search 需要评估所有 $D \times D$ 个阈值对（时间复杂度 $\mathcal{O}(ND^2)$），Frontier Search 利用 $f(\tau_t, \tau_v)$（KL 散度）和 $g(\tau_t, \tau_v)$（skipping ratio）对各自参数的单调非递减性质，将搜索复杂度降至 $\mathcal{O}(ND)$（实际约 45× 加速）。

从算法pipeline角度拆解术语：
算法流程（Alg. 1 简化版）：

```
func FrontierSearch(B[1..D], rho_target):
    # B 是排序后的候选阈值 grid: B[1] < B[2] < ... < B[D]
    frontier = []
    p = D
    for q = 1 to D:               # 遍历 tau_t 候选
        while p >= 1 and g(B[q], B[p]) >= rho_target:
            p = p - 1             # 单调递减：找到满足约束的最小 p
        p_q = p + 1               # 最小可行的 tau_v index
        if p_q <= D:
            compute f(B[q], B[p_q])  # 记录该对的目标值
            frontier.append((q, p_q))
    (q*, p*) = argmin_{frontier} f(B[q], B[p_q])
    return (B[q*], B[p*])         # 最优阈值对
```

单调性保证：
- **Monotonicity of g**: $\tau_t \uparrow$ 或 $\tau_v \uparrow$ → 更多 expert 被跳过 → $g \uparrow$。因此对于固定 q，可行的 p 集合是后缀区间 $[p_{(q)}, D]$。
- **Monotonicity of p(q)**: $q \uparrow$ → $p_{(q)} \downarrow$（$\tau_t$ 增大时，满足约束所需的最小 $\tau_v$ 减小）。因此内循环指针 p 在整个外循环中单调递减，总计至多 2D 次 guard evaluation。
- 对于每个记录到 frontier 的 $(q, p_{(q)})$ 对，只需一次前向传播计算 f。总复杂度 $\mathcal{O}(ND)$。

术语一般如何实现？如何使用？
- D=100 grid 点在 (0,1) 间等间隔采样，经 rectified sigmoid 映射。
- N=1024 calibration samples。一次 f/g 评估 = 在 calibration set 上做一次前向传播。
- 实验表明 D=100 和 D=200 的精度差异可忽略（diminishing returns），N=1024 在精度和校准成本间平衡良好。
- 搜索时间：20-30B 参数模型 < 2 hr（vs naive 搜索的 ~45 hr）。搜索时间与 D 大致线性增长。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
