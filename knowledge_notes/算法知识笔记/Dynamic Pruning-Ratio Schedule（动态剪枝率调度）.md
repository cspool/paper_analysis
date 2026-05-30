## Dynamic Pruning-Ratio Schedule（动态剪枝率调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Pruning-Ratio Schedule 是 TwigVLM++ Stage-2 RL 训练中使用的技巧，允许单个训练好的模型在推理时支持多种 pruning ratio（R̄ 值）而无需为每种 ratio 单独重训。核心思路：在 RL 训练过程中，随机化采样 R̄ 值（从候选集 R={64, 85, 107, 128, 149, 171, 192}），并使用 curriculum-based annealing 分布逐渐偏向更激进的 pruning ratio。具体来说，R̄ 的采样概率为 P(R̄=R̄_i) = exp(-β(t)·i) / Σ_j exp(-β(t)·j)，其中 β(t) = β_max · (t/T)^p 是 annealing 参数（β_max=8.0, p=2.0），t 和 T 分别是当前和总训练步数。β(t=0)=0 时分布均匀，随着训练进行 β(t→T)=β_max 时分布集中到最小 R̄ (最激进剪枝)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Dynamic Pruning-Ratio Schedule (Stage-2 RL)
R_set = [64, 85, 107, 128, 149, 171, 192]  # 候选集, 升序排列
n = len(R_set)
β_max = 8.0
p = 2.0

for t in range(T_steps):
    # 计算 annealing 参数
    β_t = β_max * (t / T) ** p
    # 采样分布
    probs = []
    for i in range(n):
        probs_i = exp(-β_t * i) / sum(exp(-β_t * j) for j in range(n))
        probs.append(probs_i)
    # 采样 R̄
    R̄ = sample_categorical(R_set, probs)
    # 用该 R̄ 进行 RL 更新
    loss = grpo_step(R̄, G=32)
    loss.backward()
    optimizer.step()
```

效果对比（RelAcc @ R̄=64/128/192）：
| RL Strategy | @R̄=192 | @R̄=128 | @R̄=64 |
|------------|--------|--------|-------|
| static R̄=192 | 99.1% | 98.8% | 97.2% |
| static R̄=64 | 99.1% | 99.0% | 98.0% |
| dynamic schedule | 99.6% | 99.2% | 97.7% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic schedule 使 P-Head 学习到在不同 R̄ 下都有效的通用剪枝策略，避免了对单一 R̄ 的过拟合（如上表：static R̄=64 在 @R̄=192 退化到 99.1%，而 dynamic 达 99.6%）。实现方式：在每个 RL training step 开始前采样 R̄，根据 R̄ 和 K, Kf 通过 Eq.(6) 反算 R 值，然后执行 GRPO-step。推理时直接指定所需的 R̄ 即可使用训练好的单一模型。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
