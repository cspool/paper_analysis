## Goal-Conditioned Markov Decision Process (Goal-Augmented MDP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Goal-Conditioned MDP（目标条件马尔可夫决策过程）是标准 MDP 的扩展，将目标 $g \in \mathcal{S}$ 作为额外输入条件注入到策略、价值函数和奖励函数中。标准 MDP 由 $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ 定义，而 Goal-Conditioned MDP 将奖励函数扩展为 $r(s_t, a_t, g): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$，策略扩展为 $\pi(a_t|s_t, g)$。这使得同一个 agent 可以泛化到训练时未见过的目标状态，而无需为每个目标重新训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Goal-Conditioned MDP 定义了 RL 训练的形式化框架。在本论文中，目标 $g = [x_g, y_g, \theta_g, 0, 0, 0]^T$ 定义为期望的位姿状态，cost 函数 $c(s,a,g) = ||W e||_{0.25}$（其中 $e = [e_{xy}, e_\theta, \beta, \dot{\beta}, v_f]^T$ 为误差向量），$p=0.25$ 使得 cost 呈稀疏特性——未收敛到目标的代价近乎恒定，只有到达目标时显著降低，从而鼓励时间最优行为。

伪代码（Goal-Conditioned RL 训练循环）：
```
# 每 episode 采样新目标
g = sample_goal()
s = env.reset()
for t = 1..T:
    a = policy(s, g)          # goal-conditioned policy
    s' = dynamics(s, a)
    c = cost(s, a, g)         # goal-conditioned cost
    buffer.store(s, a, c, s', g)
    # 从 buffer 采样更新 policy 和 critic
    update_policy_and_critic(buffer, g)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 通用实现：在 actor/critic 网络中将 goal 与 state 拼接后输入，或使用编码器将绝对坐标转换为相对目标的坐标（如本论文将 $(x_f, y_f, \theta_f)$ 编码为 $(x_f - x_g, y_f - y_g, \sin(\theta_f - \theta_g), \cos(\theta_f - \theta_g))$）。
- Hindsight Experience Replay (HER, Andrychowicz et al. 2017) 是 Goal-Conditioned RL 的关键训练技巧：将失败 episode 中实际到达的状态作为"假想目标"重新标记，大幅提高样本效率。
- 在 LLM/推理系统中，Goal-Conditioned 思想可应用于将请求的 SLO 目标作为 condition 注入调度策略。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC
