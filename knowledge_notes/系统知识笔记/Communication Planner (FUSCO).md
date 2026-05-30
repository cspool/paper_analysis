## Communication Planner (FUSCO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FUSCO 的 Communication Planner 是连接 MoE Router 的 token-level routing decisions 与 dComm 引擎的 descriptor-driven execution 的桥梁。它读取 MoE Router 产生的 token-expert 分配结果，结合集群物理通信拓扑，构建两级 descriptor plan（Node-Level + Expert-Level），供 dComm 直接执行。Planner 的关键作用是发现和利用 routing 中的 locality，消除跨节点冗余传输。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
# Planner 输入：
#   A[T × K]: token-expert matrix (router 输出)
#   expert_placement: expert_id → (node_id, gpu_id) mapping
#   token_addrs: 每个 token 在 sender tensor 中的地址和大小

# Planner 处理流程：
B = derive_token_node_matrix(A, expert_placement)  # T × N

# Step 1: Node-Level Forwarding Descriptors
for each token t:
    for each unique node n in B[t]:  # dedup: 每个 node 最多一份
        if token t not yet sent to node n:
            send_desc[t].node[n] = {addr: token_addrs[t], size: token_size}
            fwd_gpu = balancer.get_forwarder(n)  # 从 Load Balancer 获取
            recv_desc[t].node[n] = {gpu: fwd_gpu, offset: next_free_offset}

# Step 2: Expert-Level Distribution Descriptors
for each (token t, expert e) pair in A:
    node_n, gpu_g = expert_placement[e]
    local_addr = forwarder_receive_buffer[t][node_n].offset  # 在 forwarder 上的位置
    expert_offset = expert_activation_tensor[gpu_g][e].next_slot
    send_desc[t][e] = {addr: local_addr, size: token_size}
    recv_desc[t][e] = {gpu: gpu_g, offset: expert_offset}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 约 1000 行 Python，使用 PyTorch GPU operators（sum, argsort, gather, scatter）高效构建所有 metadata
- 产出 deterministic communication descriptors for each MoE layer
- 消融实验：禁用 Planner（回退到 per-token 独立通信）导致 30.2%（real-world）至 67.3%（single-node routed）的性能退化——后者因失去 token deduplication 导致跨节点通信量暴增
- Planner 在 millisecond-scale MoE communication 约束下运行，overhead 极小

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SLSQP (Sequential Least Squares Programming) 是 FSMoE 用于求解最优 pipeline degree r 的数值优化算法。FSMoE 将流水线度优化建模为 4 个带约束的单变量非线性优化问题，SLSQP 在每个 Case 中求解使耗时最小的整数 r。SLSQP 是梯度-based 的顺序二次规划方法，具有二次收敛速度。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
for c in {1,2,3,4}:
    r_c, t_c = SLSQP(minimize f_c(r), bounds=[1,inf],
                     constraints=Case_constraints)
r_opt = candidates[argmin(t_1,t_2,t_3,t_4)]
```

SLSQP 求解耗时平均 193ms（1458 配置），训练前执行一次。O(1) 额外复杂度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 使用 `scipy.optimize.minimize(method='SLSQP')`。SLSQP 源自 Nocedal & Wright "Numerical Optimization"，在 FSMoE 的单变量、行为良好的有理函数（α+β/r 或 α·r+β）场景下快速收敛。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
