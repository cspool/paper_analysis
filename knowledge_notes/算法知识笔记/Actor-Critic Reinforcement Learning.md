## Actor-Critic Reinforcement Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Actor-Critic 是强化学习中的一种混合架构，结合了 Policy-based（Actor）和 Value-based（Critic）方法的优势。Actor（策略网络 $\pi_\phi(a|s)$）直接输出动作，决定"做什么"；Critic（价值网络 $V_\psi(s)$ 或 $Q_\psi(s,a)$）评估当前策略的好坏，为 Actor 提供低方差的梯度信号。Actor 按照 Critic 的建议方向更新策略参数，Critic 则通过 TD 学习逼近真实价值函数。这种架构的优势是：Actor 可以在连续动作空间中学习（Policy Gradient 的优势），而 Critic 通过 Bootstrapping 减少梯度方差（Value-based 的优势）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在本论文中，Actor 参数化一个 Gaussian 动作分布（经 tanh 限幅后缩放至最大加速度），Critic 构造为 $L_\psi(s,a,g) = Q_\psi Q_\psi^T$（确保正输出满足 Lyapunov 下界条件）。Critic 通过最小化与 target critic $L_{target} = c + \gamma \bar{L}_{\bar{\psi}}$ 的 MSE 训练，Actor 通过最小化 $J(\phi) = \lambda_e (\log \pi_\phi + \mathcal{H}) + \lambda_l \Delta \mathcal{L}_\psi$ 训练（Lyapunov 条件违反量 + 熵正则）。

伪代码（通用 AC 算法骨架）：
```
# 初始化 Actor π_φ, Critic V_ψ
for each episode:
    s = env.reset()
    for each step:
        a = π_φ(s) + noise          # Actor: 动作选择
        s', r = env.step(a)
        δ = r + γ·V_ψ(s') - V_ψ(s)  # Critic: TD error
        V_ψ ← V_ψ + α_c·δ·∇V_ψ     # Critic 更新
        π_φ ← π_φ + α_a·δ·∇log π_φ # Actor 更新 (policy gradient)
        s = s'
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 最广泛使用的实现：stable-baselines3（PyTorch），提供 A2C、PPO、SAC、TD3 等 AC 算法。
- 本论文使用 ALAC（Lyapunov-based AC），基于 stable-baselines3 实现。Actor/Critic 均为前馈 NN（层结构 48→96→144→96→48，SoftPlus 激活），在仿真环境中训练至 $\lambda_l$ 收敛到 0.8。
- 在 AI 系统中，AC 可用于学习请求调度策略（state=队列状态，action=调度决策，reward=latency/SLO violation）。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC
