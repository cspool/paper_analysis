## Node-Link Balance Co-optimization for MoE Deployment on 3D NMP

术语解释
Node-Link Balance Co-optimization 是 HD-MoE 提出的两阶段离线自动优化框架：Stage 1 (Node Balance) 通过 Linear Programming 优化逻辑集群上的 expert 分配以平衡计算负载和减少通信量；Stage 2 (Link Balance) 通过 Bayesian Optimization 将逻辑集群映射到 2D mesh 物理节点以最小化链路级拥塞。两者分离了逻辑负载均衡和物理拓扑优化，简化搜索空间。

术语是什么？
Node-Link Balance Co-optimization 是一种将 MoE 推理部署优化问题分解为逻辑层和物理层的双层优化策略。逻辑优化（Node Balance）关注每个计算节点上的 expert 计算量和通信量是否均衡，抽象掉节点间的物理距离和拓扑约束。物理优化（Link Balance）在逻辑方案基础上，搜索最优的物理节点映射以最小化 NoC 链路拥塞。这种分离设计的动机在于：逻辑均衡的 placement 可能因为物理拓扑而产生集中的通信路径（如两个逻辑上通信量大的集群刚好在物理上相邻，链路拥塞），而直接联合优化逻辑+物理的搜索空间过大无法高效求解。

从编译框架角度拆解术语
HD-MoE Node-Link Balance Co-optimization 的完整流程：
```
输入：模型参数（E, h, IS），硬件配置（D×D mesh, comp, BW），batch size B，
      expert activation statistics（f_i, f_g from MT Bench）

# Stage 1: Node Balance (Linear Programming)
LP_Solver:
    Variables: P_ic ∈ [0,1] (连续), Z_ic ∈ {0,1} (binary indicator)
    Objective: min t_comp + 2γ·t̂_comm
    t_comp = max_c{ Σ_i P_ic·f_i·B·2h·IS / comp }
    t̂_comm = (4/BW)·max_c{ Σ_g (Π_{i∈g} Z_ic)·f_g·B·h }
    Constraints:
      Z_ic ≥ P_ic, ∀(i,c)
      Σ_c P_ic = 1, ∀i
      Σ_i P_ic·f_i ≤ (1/R_CC + 1)·ē/D, ∀c  # 负载上限
      R_CC = BW·IS·ē / (2D·comp)            # compute/comm ratio
    Output: optimal P_ic matrix (E × D)

# Stage 2: Link Balance (Bayesian Optimization)
BO_Solver:
    Search Space: all permutations of logical→physical node mapping
    Objective: min t_comm (from discrete-event simulation)
    Acquisition Function: Expected Improvement (EI)
    for iteration = 1 to N:
        propose new mapping via acquisition function
        run discrete-event simulator to evaluate t_comm
        update Gaussian Process surrogate model
    Output: best physical mapping (reduces link congestion by ~1.2×)
```
关键设计点：(1) P_ic 连续变量允许 expert 部分切分（TP 模式）；(2) 约束 R_CC 基于 compute/communication ratio 自动调整负载上限，在 compute-bound（低 BW/comp 比）时允许更大负载不均衡，在 communication-bound 时更严格约束；(3) Bayesian Optimization 适用于目标函数评估昂贵（每次需运行 DES）但相对平滑（相邻节点交换只微小影响通信成本）的场景。

术语一般如何实现？如何使用？
LP 求解可使用开源求解器（PuLP + CBC、GLPK）或商业求解器（Gurobi、CPLEX）。Bayesian Optimization 可使用 scikit-optimize、BoTorch、Ax 等框架。HD-MoE 的离线优化总耗时数小时，但只需执行一次（per model + hardware config）。代码开源：https://github.com/angerybob/HD-MoE

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
