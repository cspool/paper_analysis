## Integer Linear Programming (ILP) for Expert Placement

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Integer Linear Programming (ILP) for Expert Placement 是一种离线编译优化技术，将 MoE 模型的 expert 到 GPU 的映射问题建模为整数线性规划问题并求解。与默认的 contiguous block 放置（experts 0-1→GPU0, 2-3→GPU1...）不同，ILP 将 token 路由统计作为输入，定义二元决策变量（x_{c,e,l} 表示 expert e 是否分配 cluster c, y_{c,g,l} 表示 cluster c 是否映射 GPU g），以极小化负载不均和跨 GPU 通信为联合优化目标，通过 Gurobi 等商业求解器求得全局最优的 expert-to-GPU 映射。MoETuner 证明 ILP 在 8 experts / 4 GPUs 的 Mixtral-8x7B 规模下求解时间可控，且得到的映射在单节点和多节点均带来显著加速。

从编译框架角度拆解术语：
MoETuner 的 ILP 优化流程作为离线编译 pass，在模型部署前执行：
1. **Input IR**：Token 路由统计表（P_{e,l}: expert e 在层 l 的 token 数, R_{e_1,e_2,l}: 层 l→l+1 间 expert 对的 token 路由量），GPU 数量 G，互联带宽 B_{g_1,g_2}
2. **ILP 1 (Load-Balanced Expert Clustering)**：
   - 决策变量：x_{c,e,l} ∈ {0,1}（c ∈ [0,G-1], e ∈ [0,E-1], l ∈ [0,L-1]）
   - 目标：min Σ_{c,l} |T_{c,l} - T̄_l|，其中 T_{c,l} = Σ_e P_{e,l}·x_{c,e,l}, T̄_l = (Σ_e P_{e,l})/G
   - 约束：Σ_e x_{c,e,l} ≥ 1（每个 cluster 至少一个 expert）
3. **ILP 2 (Cluster-to-GPU Assignment)**：
   - 预计算：C_{c_1,c_2,l} = Σ_{e_1,e_2} R_{e_1,e_2,l}·x_{c_1,e_1,l}·x_{c_2,e_2,l}
   - 决策变量：y_{c,g,l} ∈ {0,1}（cluster c 是否分配到 GPU g）
   - 目标：min Σ_l max( Σ_{c_1,c_2,g_1,g_2} (C_{c_1,c_2,l} / B_{g_1,g_2})·y_{c_1,g_1,l}·y_{c_2,g_2,l+1} )
   - 约束：Σ_{l,c,e} x_{c,e,l}·y_{c,g,l} = E·L/G（每个 GPU 等量 expert），Σ_g y_{c,g,l}=1（一对一映射）
4. **Codegen/Output**：expert-to-GPU mapping tensor → PyTorch tensor 文件 → Megatron-LM 初始化时加载替换默认 placement

Gurobi 12.0.0 求解至 tolerance 0.025。ILP 使得在 GPU 数量和互联拓扑约束下，求得理论最优的 expert 放置方案。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- ILP 求解器选择：MoETuner 使用 Gurobi（商业求解器，学术界有免费 license），Alternatives: OR-Tools (Google, 开源), CBC (COIN-OR, 开源), GLPK (GNU, 开源)
- ILP 规模与可行性：Mixtral-8x7B（32 MoE layers, 8 experts/layer, 4 GPUs）→ 决策变量约 1024 个，远小于 Gurobi 可处理的上限（百万级变量）
- 与启发式方法的对比：ILP garantuees global optimality under constraints；graph partitioning / greedy heuristics 没有此保证
- 局限：ILP 求解时间为问题规模的指数函数，不直接适用于更多 experts（如 DeepSeek-V3 256 experts）——需要结合分层/迭代近似
- 类似技术在芯片设计中的应用：StreamTensor (2025) 用 ILP 做任务到 die 的分配（min inter-die communication + resource imbalance）；ExFlow (IPDPS 2024) 用 graph-based approach 而非 ILP 做 locality-aware expert placement
- MoETuner 的 ILP 依赖 token routing profiling（在目标数据集采样子集上运行推理），routing 统计表征整体行为后即可离线求解 ILP，无需在线重优化

涉及论文标题：
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
