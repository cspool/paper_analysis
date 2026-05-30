## Linear Programming for MoE Expert Placement on 3D NMP

术语解释
HD-MoE 使用 Linear Programming (LP) 将 MoE expert 在 3D NMP 节点上的最优分配问题形式化为连续优化问题。LP 同时最小化计算瓶颈（t_comp = max_c node_load）和通信量（t̂_comm 线性近似），约束条件包括 expert 完整分配、计算负载上界和资源公平性。

术语是什么？
Linear Programming（线性规划）是一种在给定线性约束条件下最小化/最大化线性目标函数的数学优化方法。HD-MoE 将 expert placement 形式化为 LP 而非 ILP（Integer Linear Programming）的关键在于使用连续变量 P_ic ∈ [0,1] 而非 binary {0,1}，这使得：(1) 求解复杂度从 NP-hard 降为多项式时间，可在数小时内处理 E×D 规模（如 64 experts × 64 nodes = 4096 变量）；(2) 连续分配允许单 expert 部分切分到多节点，实现了 TP-EP 的自然混合。

从编译框架角度拆解术语
HD-MoE 的 LP 形式化包含以下组件：
```
Minimize: t_node_overhead = t_comp + 2·γ·t̂_comm
Subject to:
  (C5) Z_ic ≥ P_ic          # binary indicator constraint
  (C6) Σ_c P_ic = 1          # expert 完整分配
  (C7) t_comp ≥ per-node computation time  # 取 max
  (C8) t̂_comm ≥ per-node communication volume  # 取 max  
  (C9) Σ_i P_ic·f_i ≤ (1/R_CC + 1)·ē/D   # 负载上界
  (C10) R_CC = BW·IS·ē/(2D·comp)          # compute/comm ratio
```
约束 C9/C10 的设计思想：R_CC = TP 模式下的理论 compute/comm 时间比。若节点负载超过 TP baseline 的 (1/R_CC+1) 倍，则该节点同时是计算和通信瓶颈，此 placement 不可能优于 TP baseline，故可提前剪枝。γ 系数通过 DES 经验校准（R² > 0.9），使线性通信近似 t̂_comm 能够准确反映实际 DES 测量的不规则 all-to-all 延迟。

术语一般如何实现？如何使用？
LP 求解器的选择影响求解效率：PuLP（Python, 支持 CBC/GLPK, 开源免费）、Gurobi/CPLEX（商业, 速度更快）、ortools（Google, 开源）。HD-MoE 论文未明确指定使用的 LP solver。对于 E=64, D=64 的规模，变量数 ~4096 (P_ic) + 4096 (Z_ic)，约束数 ~10000，使用 Gurobi 可在分钟级完成。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
