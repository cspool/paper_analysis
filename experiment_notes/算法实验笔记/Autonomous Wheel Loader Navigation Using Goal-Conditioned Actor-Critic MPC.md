## Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

> ⚠️ 注意：此论文目录名为 "KTransformers Unleashing the Full Potential of CPUGPU Hybrid Inference for MoE Models"，但 PDF 实际内容为轮式装载机自主导航的机器人控制论文（ICRA 2025，arXiv:2409.15717）。按"提出新的算法模型"归类为算法pipeline 层次，为弱匹配。

- 属于算法pipeline的实现是什么？实验比较什么？
  - 论文提出 **Goal-Conditioned Actor-Critic MPC**：将 Lyapunov-based Actor-Critic RL（ALAC 算法）训练的 critic 网络作为非线性 MPC 的 terminal cost 和 stage cost，替代传统的高层轨迹规划器。核心算法组件：
    1. **Lyapunov-based RL 训练（ALAC）**：训练 critic L_ψ 满足 sampling-based Lyapunov 稳定性条件（Theorem 3.1）。引入 gradient penalty（Eq. 16）鼓励 1-Lipschitz，为下游 MPC 提供平滑优化景观。
    2. **Critic 作为 Terminal Cost**：l_f(x_N,g) = L_ψ(x_N, 0, g)，动作替换为零向量。
    3. **Critic 二阶 Taylor 近似作为 Stage Cost**：l(x_n,u_n,g) = Δt · L̃(x_n,u_n,g)，L̃ 为 critic 在上一 MPC 解处的二阶 Taylor 展开（Eq. 25），缓解仅用 terminal cost 时的犹豫行为。
    4. **输入延迟补偿**：将 MPC 初始状态向前传播 200ms 匹配执行器延迟。
  - 实验比较：
    - **Baseline**：基于 CasADi + IPOPT 的非线性轨迹优化（Eq. 27-28），T=25s horizon，direct collocation 离散化（采样 200ms），求解 >5s（AMD Ryzen 3900x）。
    - **场景 (a)/(b)**：真机实验（Avant 635），短装载循环和 180° 紧凑转弯，收敛时间与 baseline 相当或更优。
    - **场景 (c)**：多障碍物导航仅仿真（N=20 使 MPC 求解 200-300ms > 实时 100ms 要求）。
    - **128 场景仿真**：Actor-Critic MPC 平均收敛 10.92s vs baseline 14.33s（快 23.80%），全部成功。
    - **指标**：收敛时间（||x-g|| < 0.1）、速度跟踪误差。

- 硬件平台是什么，配置是什么。
  - **真机**：NVIDIA Jetson AGX Orin（32GB unified memory，12 cores），搭载于 Avant 635 小型轮式装载机。
  - **Baseline**：桌面 AMD Ryzen 3900x CPU（离线求解，>5s）。
  - **执行器**：液压转向 + 柴油发动机，输入延迟约 200ms。

- 模型是什么。数据集和bench分别是什么。
  - **Actor/Critic 网络**：前馈 NN，层结构 (48,96,144,96,48)，SoftPlus 激活。Critic 构造 L=Q·Q^T（确保正输出）。编码器将绝对位姿转为相对位姿，heading 用 sin/cos 编码。Actor 为 Gaussian policy（tanh + 缩放至 a_max）。
  - **运动学模型**：6D 状态 [x_f, y_f, θ_f, β, β̇, v_f]，2D 控制 [β̈, a_f]。4 阶 Runge-Kutta 离散化（Δt=0.2s）。
  - **场景**：无标准 benchmark。真机：(a) 短装载循环，(b) 180° 转弯。仿真：(c) 多障碍物场景，128 随机目标位姿。
  - **RL 训练**：PyTorch + stable-baselines3。MPC：CasADi + Acados + L4CasADi（NN 集成），SQP-RTI + HPIPM QP solver，N=10（N=20 for obstacles）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **未找到公开代码仓库**（搜索 arXiv:2409.15717 + "Aleksi Mäki-Penttilä" + GitHub 无匹配）。
  - **算法 pipeline 伪代码**：

```
=== 阶段一：RL 离线训练（仿真环境） ===
Input: 运动学模型 f(x,u), 目标 goal g
Output: 训练好的 critic L_ψ (满足 Lyapunov 条件)

1. 初始化: Actor π_φ, Critic L_ψ = Q_ψ·Q_ψ^T
2. for each episode:
3.     s0 ← 重置; g ← 采样目标
4.     for each step t:
5.         a_t ~ π_φ(s_t, g)              # Gaussian policy + tanh 限幅
6.         s_{t+1} = f(s_t, a_t)           # 4阶 Runge-Kutta, Δt=0.2s
7.         存储 (s_t,a_t,c_t,s_{t+1},g) → replay buffer D
8.         从 D 采样 mini-batch
9.         # Critic 更新 (Eq. 12-16):
10.        L_target = c(s,a,g) + γ·L̄_ψ̄(s', a', g)
11.        Ĵ_c = E[(L_ψ - L_target)²] + ρ·E[(1-||∇L_ψ||₂)²]
12.        # Actor 更新 (Eq. 17-18):
13.        J_φ = λ_e·(log π_φ + H) + λ_l·ΔL_ψ
14.        # Lagrange multipliers 自适应 (Eq. 19):
15.        当 λ_l < 1 → L_ψ 为有效 Lyapunov 函数
16. 训练至 λ_l 收敛 → 0.8 (实验设定)

=== 阶段二：MPC 在线求解（真机 Jetson，每步 <100ms） ===
Input: 当前状态 x_init, 目标 g, 训练好的 critic L_ψ

1. 输入延迟补偿: x_init ← propagate(x_init, 200ms) via Eq. 5
2. 构建 NLP (Eq. 20):
   min_{x,u} Σ_{n=0}^{N-1} [Δt · L̃(x_n,u_n,g)] + L_ψ(x_N,0,g)
   s.t. x_{i+1}=f(x_i,u_i), 状态约束 (Eq. 6), 控制约束 (Eq. 21)
        可选: 障碍物约束 (Eq. 22)
3. Stage cost L̃: critic 在上一解 (x*,u*) 处的二阶 Taylor 近似 (Eq. 25)
   设 z_n=[x_n;u_n], z*_{n+1}=[x*_{n+1};u*_{n+1}]:
   L̃(x_n,u_n,g) = ∂L/∂z_n · (z*_{n+1}-z_n) + 0.5 ∂²L/∂z_n² · (z*_{n+1}-z_n)²
4. Warm-start SQP-RTI solver (HPIPM QP solver)
5. solver → 最优轨迹 [x₀..x_N], 取 x₁ 的 (β̇,v_f)
6. 发送 (β̇_cmd, v_f_cmd) → 低层反馈控制器 → 液压/发动机
```

  - **张量计算流**（单 MPC 迭代，N=10）：
    1. 编码：绝对位姿 → 相对位姿 (dx, dy, sin(θ-θ_g), cos(θ-θ_g))
    2. Terminal: x_N → NN [48→96→144→96→48] SoftPlus → Q → L=Q·Q^T → scalar
    3. Stage: 在 (x*,u*) 处计算 ∂L/∂z 和 ∂²L/∂z²（各 N 步），构造二次型（比直接 NN forward 更轻量）
    4. SQP-RTI: 序列二次规划 → 输出 x₁ 的 β̇, v_f

> ⚠️ **近似层次匹配说明**：此论文属于机器人控制/AI 决策领域，非典型 AI 系统/LLM 推理领域。因"提出新的算法模型"（Goal-Conditioned Actor-Critic MPC）归类为算法pipeline，为弱匹配。其他层次（Serving调度/编译框架/kernel调度/硬件架构/芯片设计）均不适用。论文实验基于 kinematics simulation + 真机测试（Avant 635 装载机），硬件为 NVIDIA Jetson AGX Orin，无 GPU 集群或 AI 加速器。
