## Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

> ⚠️ 目录名为 "KTransformers..." 但 PDF 实际为轮式装载机自主导航论文（ICRA 2025, arXiv:2409.15717），属于机器人控制/AI 决策领域。按 "提出新的算法模型" 归入此库，为弱匹配。

- baseline方法是什么？
  - **传统高层轨迹规划器 + MPC 跟踪控制**：Baseline 采用两步法：
    1. 高层轨迹规划器生成参考轨迹：RRT* based planner [22] 或 optimization-based planner [23] 离线/非实时生成全局路径 → LPV-MPC 或 adaptive MPC 作为参考轨迹跟踪控制器。
    2. 直接轨迹优化（论文 baseline）：使用 CasADi + IPOPT 求解非线性轨迹优化问题（Eq. 27-28），T=25s horizon, direct collocation 离散化（采样 200ms），目标函数为 p-norm 形式的误差积分。求解时间 >5s（AMD Ryzen 3900x desktop CPU），无法实时运行。
  - **Baseline 缺陷**：
    1. **非实时 planners 导致次优性**：离线规划器（如 optimization-based planner in [23]）生成轨迹后，因建模误差实际执行会出现偏离，而 planner 无法及时更新 → 沿过时轨迹跟踪导致次优。实时 planners（如 RRT* in [22]）通过采样/离散化保证实时性，但轨迹高度次优。
    2. **短 horizon MPC 缺乏规划能力**：MPC 若单独使用（无高层 planner），受限于预测 horizon（如 N=10, 2s），无法求解需要远见（>10s ahead）的复杂规划任务。baseline 轨迹优化用 T=25s 但无法实时（>5s 求解时间）。
    3. **RL actor 无法直接部署**：纯 RL policy 在仿真中可完成任务，但无法考虑执行器限幅和其他约束，直接部署不安全。
  - 全栈执行例子（Baseline 轨迹优化，Eq. 27-28）：
    - **模型推理/训练算法层**：构建 NLP 目标函数 ∫[0→T] ∜e(t) + β² + β̈² + a_f² dt，使用 direct collocation + IPOPT 求解 → 输出全程轨迹。论文未明确说明。
    - **系统框架层**：CasADi（符号优化框架）→ IPOPT（内点法 NLP solver）。论文未明确说明其他系统级组件。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明。CasADi/IPOPT 内部数值计算（矩阵分解等）为标准 CPU 计算。
    - **硬件架构层**：AMD Ryzen 3900x desktop CPU。离线求解，无实时性要求。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Goal-Conditioned Actor-Critic MPC**：将 RL 训练出的 critic 作为 MPC 的 cost function，使短 horizon MPC 继承 RL agent 的长期规划知识。
  - **核心设计选择与 Baseline 缺陷的直接映射**：
    1. **Critic as Terminal + Stage Cost**：RL critic L_ψ(s,a,g) 经离线训练后编码了"从当前状态到 goal 的 long-term optimal value"（Eq. 15: 折扣无限 horizon 期望 cost 和）。作为 terminal cost 时，即使 MPC horizon N=10 (2s)，也能通过 terminal cost 评估后续无限 horizon 的代价 → 等价于"MPC 有隐式长 horizon 规划能力"。Stage cost 通过二阶 Taylor 近似（Eq. 25）提供逐步引导，解决"非终端阶段犹豫行为"。
    2. **Lyapunov 稳定性保证**：使用 ALAC 算法训练 critic 满足 sampling-based Lyapunov 条件（Eq. 3a-3c）→ critic 值沿系统轨迹递减，保证 mean cost stability → MPC 继承稳定性。
    3. **Gradient Penalty for MPC 优化**：在 critic loss 中加入 ρ·E[(1-||∇L_ψ||₂)²]（Eq. 16），鼓励 critic 为 1-Lipschitz → 为下游 SQP-RTI solver 提供更平滑的优化景观，缓解前人工作中 "因 NN critic 高度非线性导致优化困难" 的问题 [17]。
    4. **输入延迟补偿 + 约束强制执行**：MPC 将初始状态向前传播 200ms 补偿执行器延迟。MPC 强制状态/控制约束（Eq. 6, 21），克服纯 RL actor 无法处理约束的问题。
  - 对应解决 Baseline 缺陷：
    - Baseline 非实时 planner → method 将 planner 知识编码到 critic 中，MPC 每步 <100ms（Jetson），实时。
    - Baseline 短 horizon MPC 缺乏规划能力 → critic 提供 implicit 长 horizon 规划（terminal cost 等价于无限 horizon value）。
    - Baseline RL 不安全 → MPC 强制执行约束（状态/控制/障碍物）。
    - Baseline 轨迹优化 >5s → MPC solver <100ms per iteration（10-20× faster on weaker hardware）。
  - 全栈执行例子（Actor-Critic MPC，Avant 635 真机）：
    - **模型推理/训练算法层**：离线 RL 训练（PyTorch + stable-baselines3）→ 在线 MPC（CasADi + Acados + L4CasADi）。每 200ms 迭代：编码状态 → Critic forward（terminal cost）+ Taylor 近似（stage cost）→ SQP-RTI 求解 NLP → 输出 β̇, v_f 给低层控制器。
    - **系统框架层**：CasADi（NLP 建模）+ Acados（SQP-RTI solver）+ HPIPM（QP solver）+ L4CasADi（NN 集成到 CasADi 符号表达式）。低层控制器跟踪 MPC 输出的速度指令。论文未明确说明操作系统级框架。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：论文未明确说明。NN 推理（PyTorch on Jetson GPU）和 SQP solver（CPU）并行在 Jetson AGX Orin 上运行。
    - **硬件架构层**：NVIDIA Jetson AGX Orin（嵌入式平台，ARM CPU + NVIDIA GPU on SoC）。真机液压/柴油执行器延迟约 200ms。Baseline 用 desktop AMD Ryzen 3900x。
  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | Actor-Critic MPC 设计 | 效果 |
    |-------------|----------------------|------|
    | 非实时 planner (>5s) | Critic 编码规划知识 → MPC 每步 <100ms | 实时控制 |
    | 短 horizon 缺乏规划 | Terminal cost = critic (无限 horizon value) | N=10 等价长 horizon 规划 |
    | RL actor 不安全 | MPC 强制约束 (Eq. 6, 21, 22) | 安全真机部署 |
    | NN critic 难以优化 (前人工作) | Gradient penalty (1-Lipschitz) | 平滑 MPC 优化景观 |
    | 执行器延迟 200ms | 前向传播初始状态 200ms | 延迟补偿 |
    | 无逐步指导（前人工作仅 terminal cost） | 二阶 Taylor stage cost (Eq. 25) | 消除犹豫行为 |
