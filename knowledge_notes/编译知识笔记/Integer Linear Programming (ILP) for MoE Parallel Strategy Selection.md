## Integer Linear Programming (ILP) for MoE Parallel Strategy Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Integer Linear Programming (ILP) for MoE Parallel Strategy Selection 是 HAP 提出的将 MoE 推理的并行策略选择问题形式化为整数线性规划问题的方法。ILP 目标是最小化端到端推理延迟：min T_total = T_prefill + T_decoding + C_ij（策略切换开销）。决策变量为 one-hot 向量 S_k（Attention 模块选择第 k 种策略）和 E_i/E_j（Expert 模块在 prefill/decode 分别选择第 i/j 种策略）。约束包括：(1) 内存约束——KV cache + Attention 权重 + Expert 权重 + activation 内存 < per-GPU 显存；(2) 并行度约束——A_t × A_d = E_d × E_t × E_e = N（总设备数）；(3) 整除约束——隐藏维度、KV head 数、expert 数、expert 中间维度必须被对应并行度整除。HAP 使用 Python PuLP 库求解 ILP，在典型 8-GPU 配置下 <1 秒完成。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

ILP 求解在 HAP 推理初始化阶段执行，作为"编译"步骤决定运行时并行策略：

```
# HAP ILP 求解流程 (PuLP-based)
# 输入: 模型配置 + 硬件配置 + 仿真模型输出

# Step 1: 构建搜索空间
search_space = {
    "Attention": ["DP", "TP", "DP+TP"],  # K_a 种策略
    "Expert": ["EP", "TP", "EP+TP"],       # K_e 种策略 (排除DP)
}

# Step 2: 枚举可行策略组合，过滤违反内存约束的组合
feasible_attn = []
for strategy in search_space["Attention"]:
    (A_t, A_d) = get_parallel_degrees(strategy, N)
    if memory_check(A_d, M_attn, M_KV, M_gpu):
        feasible_attn.append((strategy, A_t, A_d))

feasible_expert = []
for strategy in search_space["Expert"]:
    (E_t, E_d, E_e) = get_parallel_degrees(strategy, N)
    if memory_check(E_d, M_exp, M_act, M_gpu, conservative=True):
        feasible_expert.append((strategy, E_t, E_d, E_e))

# Step 3: 构建 ILP 问题
prob = LpProblem("Minimize_Inference_Latency", LpMinimize)

# 决策变量 (one-hot selection)
S = [LpVariable(f"S_{k}", cat="Binary") 
     for k in range(len(feasible_attn))]
E_prefill = [LpVariable(f"E_prefill_{i}", cat="Binary") 
             for i in range(len(feasible_expert))]
E_decode = [LpVariable(f"E_decode_{j}", cat="Binary") 
            for j in range(len(feasible_expert))]

# 约束: 每模块只选一种策略
prob += lpSum(S) == 1
prob += lpSum(E_prefill) == 1
prob += lpSum(E_decode) == 1

# 目标函数:
# N_layer × [prefill_latency + S_output × decode_latency] + switching_cost
prefill_term = lpSum([
    S[k] * T_attn_k + E_prefill[i] * T_expert_i + T_comm_ki
    for k in ... for i in ...
])
decode_term = lpSum([
    S[k] * T_attn_k + E_decode[j] * T_expert_j + T_comm_kj
    for k in ... for j in ...
])
switching_term = lpSum([
    E_prefill[i] * E_decode[j] * C_ij
    for i in ... for j in ...
])

prob += N_layer * (prefill_term + S_output * decode_term) + switching_term

# Step 4: 求解
prob.solve(PULP_CBC_CMD(msg=False))  # <1 second typical
```

ILP 的关键简化：(1) Attention 模块在 prefill 和 decode 使用相同策略（因 KV cache 约束）；(2) Expert 模块排除 DP（因权重占 90% 参数，复制到多 GPU 显存不足）；(3) 排除 DP+EP+TP 三合一组合（单节点多 GPU 场景下性能不最优）。内存约束对 EP 采用保守估计：activation 内存按 2× TP activation footprint 上限估算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HAP 使用 Python PuLP (https://github.com/coin-or/pulp) 开源 ILP 求解库，默认使用 CBC 求解器。ILP 求解的输入参数（T_attn, T_expert, T_comm, C_ij）来自随机森林回归仿真模型。搜索空间枚举在推理前完成（一次性），ILP 求解也在推理前完成。运行时直接使用 ILP 输出的最优策略配置，无需重复求解。

涉及论文标题：
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
