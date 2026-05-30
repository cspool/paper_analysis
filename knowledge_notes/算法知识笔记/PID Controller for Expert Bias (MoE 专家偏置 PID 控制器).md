## PID Controller for Expert Bias (MoE 专家偏置 PID 控制器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PID Controller for Expert Bias 是 LongCat-Flash 在 aux-loss-free load balancing [Wang et al., 2024a] 基础上提出的改进。核心思想：使用比例-积分-微分 (PID) 控制器动态调整 expert bias $b_i$，使每个 FFN expert 的 token 分配精确收敛到目标比例，同时不干扰 LM 训练目标。相比原方案使用的固定 bias increment，PID 控制器提高了 softmax router 在大规模 expert 数量下的概率分布鲁棒性。

Bias 更新公式：$$\Delta b_i = \begin{cases} \mu \left( \frac{K_e}{K} \cdot \frac{1}{N} - \frac{T_i}{K T_{\text{all}}} \right), & 1 \le i \le N \\ 0, & N < i \le N + Z \end{cases}$$ 其中 $\mu$ 为 bias adaptation rate（decay schedule），$T_{\text{all}}$ 为 global batch 的 token 总数，$T_i$ 为路由到 expert i 的 token 数，$K_e$ 为期望的 FFN expert 激活数。关键设计：(1) Zero-computation experts 的 bias 固定为零（不参与更新），因为它们的 identity 性质只需要全局约束——当所有 FFN experts 达到目标比例时自动满足；(2) 大 batch size + decay schedule for $\mu$ 提高 budget control 的稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# PID Expert Bias Update (每 training step)

参数:
  N: 512 FFN experts, Z: 256 zero-comp experts
  K: 12 (top-K), K_e: 8 (期望 FFN 激活数)
  mu: bias adaptation rate (根据 global batch size 和 schedule 衰减)
  b: [N+Z]  # expert bias, 初始化为 0

# 每个 step 执行:
T_all = 0  # global batch 总 token 数
T = [0] * (N+Z)  # 每个 expert 接收的 token 数

# 收集统计 (跨所有 EP group)
for each micro_batch:
    for each token x_t:
        probs = softmax(router(x_t) + b)  # 加上当前 bias
        topk_indices = topk(probs, k=K)
        for idx in topk_indices:
            T[idx] += 1
            T_all += 1

# 更新 bias (仅 FFN experts)
for i in range(N):
    target_ratio = (K_e / K) * (1.0 / N)  # 每 expert 目标占比
    actual_ratio = T[i] / (K * T_all)
    delta = mu * (target_ratio - actual_ratio)
    b[i] += delta

# zero-comp experts (N 到 N+Z-1): b 保持 0，不更新
```

LongCat-Flash 观察：大 batch size 和 μ 的 decay schedule 提高收敛稳定性；小 batch size 可能需要降低更新频率。经过约 20B tokens 训练后所有层平均 expert 数收敛至目标值（波动 <1%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 原方案 [Wang et al., 2024a] 使用固定增量的 bias 更新，PID 改进在于用 error 比例驱动更新而非固定步长。
2. μ 使用 decay schedule（如 cosine decay），从较大初始值开始确保快速收敛，后期降低避免振荡。
3. 与 device-level load balance loss (Eq. 3-5) 互补：PID 控制 corpus-level 平均负载，load balance loss 防止 sequence-level 极端不均衡。
4. 依赖大 global batch size——LongCat-Flash 使用 tens of thousands of accelerators，global batch 足够大以保证统计稳定。

涉及论文标题：
- LongCat-Flash Technical Report
