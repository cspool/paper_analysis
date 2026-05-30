## Hybrid TP-EP Parallelism for MoE on 3D NMP

术语解释
Hybrid TP-EP Parallelism 是 HD-MoE 提出的一种将 Tensor Parallelism (TP) 和 Expert Parallelism (EP) 混合使用的自动并行策略，专为 3D NMP 分布式架构设计。与 GPU 集群上的 Hybrid TP-EP（将 mesh 划分为子区域，区域内 TP + 区域间 EP，通过复制 hot expert 缓解不均衡）不同，HD-MoE 的 Hybrid TP-EP 允许**单个 expert 在不同节点间部分切分**（连续变量 P_ic ∈ [0,1]），high-frequency expert 使用 TP 模式分担计算负载，low-frequency expert 使用 EP 模式避免通信开销。

术语是什么？
Hybrid TP-EP Parallelism 是一种结合两种并行策略的 MoE 推理部署方法：(1) Tensor Parallelism (TP)：将单个 expert 的权重沿 intermediate dimension 切分到多个节点，各节点计算部分输出后通过 all-reduce 聚合（计算均衡，通信开销大）；(2) Expert Parallelism (EP)：将完整 expert 分配给单一节点，token 通过 all-to-all 路由到对应节点（通信少，负载不均衡）。HD-MoE 的关键创新是将 expert 分配形式化为连续变量 P_ic（expert i 在节点 c 的分配比例），使得 hot expert 可以同时分配到多个节点（TP 模式），cold expert 保持完整分配（EP 模式），且这一分配由 LP 求解器自动搜索得到。

从kernel调度角度拆解术语
HD-MoE Hybrid TP-EP 的执行伪代码：
```
# 输入：LP 求解的连续 placement matrix P_ic (E x D)
# 输入：token-to-expert activation (B tokens, each activates e experts)
# 输入：logic cluster to physical node mapping (from BO)
for each MoE layer:
    # 1. Token Dispatch (All-to-All)
    for each token t:
        for each activated expert e_i in t.experts:
            # 选择持有 expert i 的物理节点
            candidates = {c | P_ic > 0}  # 多个节点可能持有部分 expert
            target = argmin_c(load[c]) among candidates
            send(t.hidden_state, src=current_node, dst=target)
    
    # 2. Expert Computation (per node)
    for each node c in parallel:
        assigned_tokens = received_tokens[c]
        for each expert i where P_ic > 0:
            # 如果 P_ic = 1: 完整 EP 模式，本地计算完整 FFN
            # 如果 0 < P_ic < 1: TP 模式，计算 1/|holders| 的中间维度
            partial_output = expert_i.ffn(assigned_tokens, slice=P_ic)
        # 3. Result Aggregation
        for each token t:
            if expert i has P_ic < 1:  # TP 模式需要 all-reduce
                all_reduce(partial_output, group=holders_of_expert_i)
```
关键设计：P_ic 连续值允许 hot expert (高 f_i) 部分切分以平衡计算（多个节点分担），cold expert (低 f_i) 完整分配以避免通信。LP 目标函数 min(t_comp + 2γ·t̂_comm) 同时优化计算均衡和通信量。

术语一般如何实现？如何使用？
GPU 集群上的 Hybrid TP-EP（如 DeepSeek-V3 部署 DeepSeek-R1）采用 EP + hot expert replication 方式，即 EP 为主要策略，但将高频 expert 复制到多个节点以避免负载不均衡。但这在 3D NMP 上不适用（内存受限无法复制完整 expert）。HD-MoE 的连续 P_ic 方案是 3D NMP 特化的实现，通过 LP 求解器（如 PuLP、Gurobi、CPLEX）离线搜索最优 P_ic。代码开源：https://github.com/angerybob/HD-MoE

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
