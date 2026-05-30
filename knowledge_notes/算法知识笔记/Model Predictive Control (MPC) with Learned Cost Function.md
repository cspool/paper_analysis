## Model Predictive Control (MPC) with Learned Cost Function

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model Predictive Control (MPC) 是一种基于模型的优化控制方法。在每个控制周期，MPC 求解一个有限 horizon 的约束优化问题（通常为 NLP），得到最优控制序列，但仅执行第一步控制，下一周期重新求解（receding horizon）。标准 MPC 的 cost function 通常由人工设计（如二次型跟踪误差），而本论文中的 Actor-Critic MPC 使用 RL 训练的 critic $L_\psi$ 作为 cost function——terminal cost $l_f(x_N, g) = L_\psi(x_N, 0, g)$，stage cost $l(x_n, u_n, g) = \Delta t \cdot \tilde{L}(x_n, u_n, g)$（critic 在上一解处的二阶 Taylor 近似）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

本论文的 Actor-Critic MPC 每步执行流程（Jetson AGX Orin, <100ms/iter）：
```
1. 输入延迟补偿: x_init ← propagate(x_init, 200ms)  # 匹配执行器延迟
2. 构建 NLP (Eq. 20):
   min Σ_{n=0}^{N-1} Δt·L̃(x_n, u_n, g) + L_ψ(x_N, 0, g)  # stage + terminal cost
   s.t. x_{i+1}=f(x_i, u_i)  # 运动学模型 (4阶 Runge-Kutta)
        状态/控制/障碍物约束
3. Stage cost L̃: critic 在上一解处的二阶 Taylor 近似
   L̃(z_n) ≈ ∂L/∂z·(z* - z_n) + 0.5 ∂²L/∂z²·(z* - z_n)²
   （比直接 NN forward 更轻量）
4. SQP-RTI solver (HPIPM QP) → 最优轨迹
5. 取 x₁ 的 (β̇, v_f) → 发送给低层控制器
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 标准 MPC 工具链：CasADi（符号建模）+ Acados（嵌入式优化框架）+ HPIPM（高性能 QP solver）。本论文使用 CasADi + Acados + L4CasADi（将 PyTorch NN 转换为 CasADi 符号表达式）。
- Learned Cost MPC 的关键挑战：(a) NN 的高度非线性使 NLP 难以优化 → 通过 gradient penalty (1-Lipschitz) 缓解；(b) NN 推理开销 → 通过 Taylor 近似（仅需前向一次 NN 计算 terminal cost，stage cost 复用）。
- 在 LLM 推理系统中，MPC 可用于动态调整 batch size、request admission control 等资源分配问题（如以 latency 和 throughput 为 cost，GPU memory 和 compute 为约束），但目前尚无相关工作将 Learned Cost MPC 用于 LLM serving。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC
