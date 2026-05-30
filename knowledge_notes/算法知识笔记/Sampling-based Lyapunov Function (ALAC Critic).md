## Sampling-based Lyapunov Function (ALAC Critic)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sampling-based Lyapunov Function 是 Adaptive Lyapunov-based Actor-Critic (ALAC, Wang et al. 2023) 算法训练的 Critic 网络所满足的稳定性证书。传统 Lyapunov 函数要求对整个状态空间严格满足衰减条件，而 Sampling-based Lyapunov 仅要求在经验采样分布 $\mathcal{S}_\pi$ 上满足。具体条件（Theorem 3.1）：存在 $L(s,g)$ 满足 (a) $k_l c_\pi \le L \le k_u c_\pi$（上下界），(b) $L(s,g) \ge c_\pi + \lambda \mathbb{E}[L(s',g)]$（期望衰减），(c) 特定不等式约束在稳态分布上成立。当 Critic 满足这些条件时，系统满足 mean cost stability（Definition 3.1: $\lim_{t\to\infty} \mathbb{E}[c_\pi(s_t,g)] = 0$）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ALAC 训练过程中，Critic 通过最小化 Bellman-like loss $J_c(\psi) = \mathbb{E}[(L_\psi - (c + \gamma \bar{L}_{\bar{\psi}}))^2]$ 来逼近 sampling-based Lyapunov function。Actor 则通过 Lagrange 乘子 $\lambda_l$ 引导 Critic 满足条件 (c)。当 $\lambda_l$ 自适应下降到 < 1 时（本论文训练至 0.8），Critic 被认证为有效 Lyapunov 函数。

Lyapunov 条件验证流程：
```
训练循环中:
1. Critic target: L_target = c(s,a,g) + γ·L̄(s', a', g)  // TD-style bootstrap
2. Critic loss: Ĵ_c = E[(L - L_target)²] + ρ·E[(1-||∇L||₂)²]  // + gradient penalty
3. 计算条件(c)违反量: ΔL = L(s',π(s')) - L(s,a) + k(L(s,a) - λL(s',π(s')))
4. Actor loss: J_φ = λ_e·(log π_φ + H) + λ_l·ΔL    // 引导满足条件(c)
5. Lagrange 更新: λ_l ← λ_l + α·ΔL                 // 收敛至0.8→Lyapunov有效
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- ALAC 论文（Wang et al. 2023, CoRL）开源: https://github.com/ShengjieWang00/ALAC
- 与标准 Actor-Critic 的关键区别：Critic 构造为正定形式 $L=QQ^T$（满足条件a），额外的 Lagrange 乘子自适应机制保证条件(c)。
- 在 LLM 推理系统中的应用潜力：将 Lyapunov 稳定性用于保证请求队列的 long-term stability（mean latency bounded），但尚未有相关工作。
- 本论文创新：在 critic loss 中加入 gradient penalty 项 $\rho\mathbb{E}[(1-||\nabla L||_2)^2]$（Eq. 16），鼓励 critic 的 1-Lipschitz 性质，缓解下游 MPC 优化困难。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC
