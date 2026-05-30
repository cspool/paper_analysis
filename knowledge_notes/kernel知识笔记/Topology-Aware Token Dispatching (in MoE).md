## Topology-Aware Token Dispatching (in MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Topology-Aware Token Dispatching 是 Hecate 的 Dispatcher 组件使用的 token 路由策略。当 sparse materialization 使同一个 expert 的参数在多个 device 上有副本时，每个 token 需要从多个候选 target device 中选择一个来 dispatch。Hecate 的 dispatching 算法优先 intra-node 通信路径：若 token 的 source device 所在 node 内有该 expert 的 replica，则优先选择该 node 内的 device；仅当 node 内无 replica 时才跨 node dispatch。当有多个候选 device 时，均匀分配 tokens 以平衡负载。

这种拓扑感知策略减少了跨 node 的 All-to-All 通信量，因为 inter-node 带宽（如 100 Gbps NIC）通常远低于 intra-node 带宽（如 600 GB/s NVSwitch）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Topology-Aware Dispatching (Hecate Dispatcher):
-------------------
输入:
  routes: token → expert assignment (来自 MoE gate)
  P': 当前 materialized expert placement
       (哪些 expert 在哪些 device 上有副本)
  topology: node/device 映射

输出: dispatch_plan: 每个 token → 目标 device

for each token t on source device d_src:
    expert_e = routes[t]
    candidates = {d | (expert_e, d) in P'}  // expert 的所有 replica device

    // Priority 1: 同 device
    if d_src in candidates:
        dispatch_plan[t] = d_src  // 本地计算, 零通信

    // Priority 2: 同 node (intra-node NVLink/NVSwitch)
    else if exists d_candidate in candidates where node[d_candidate] == node[d_src]:
        // 均匀分配到同 node 的候选 devices
        dispatch_plan[t] = least_loaded(candidates_in_same_node)

    // Priority 3: 跨 node (inter-node NIC)
    else:
        // 均匀分配到所有候选 devices
        dispatch_plan[t] = least_loaded(all_candidates)

// 效果: 最小化跨 node All-to-All 通信量
// Hecate 实验中 A2A 时间相比 EP 减少 12.3×
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 Hecate 的 Executor 中，MoE gate 输出 token assignment 后调用 Dispatcher 生成 dispatching plan。
- Dispatcher 需要 topology map（node-device 隶属关系），在 cluster setup 时建立。
- 均匀分配（least_loaded）防止同一 expert 的所有 tokens 涌入同一 device 造成新的 straggler。
- 拓扑感知 dispatching 结合 sparse materialization 的拓扑感知 placement search（Algorithm 1 优先 intra-node placement），两者协同减少跨 node 通信。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
