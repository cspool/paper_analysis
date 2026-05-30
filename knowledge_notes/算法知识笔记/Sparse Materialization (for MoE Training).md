## Sparse Materialization (for MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse Materialization 是 Hecate 在 FSSDP 的 materialization phase 中使用的 expert placement 搜索算法 (Algorithm 1)。其目标是在两个系统约束下搜索近似最优的 expert placement：(1) overlap degree t——可在 attention computation 时间内隐藏通信的 expert 物化数上限；(2) memory capacity m——每 device 可额外容纳的 expert 参数数。基于预测的 expert load distribution（滑动窗口平均，w=5），算法决定哪些 expert 需要 replica（物化到多个 device）以及 replica 的分布。

算法是拓扑感知的：overlap degree t 的计算使用 inter-node bandwidth（异构网络时）或 uniform bandwidth（同构网络时），优先避免跨 node 通信。当 t > m 时（高 overlap degree 但内存受限），按 expert load 比例分配 replica slots，优先有空闲 slots 的 node。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Algorithm 1: Sparse Materialization
Input: P (sharded placement), F (expert load distribution),
       t (overlap degree), m (memory capacity)
Output: P' (materialization plan)

t ← min(t, |E|), m ← min(m, t)
P' ← P

if t <= m:
    // Case 1: 内存充裕，可将 top-t expert 物化到所有 device
    E^topT ← Top t experts by load F
    P' ← P' ∪ (D × E^topT)  // 所有 device 都接收这些 expert
else:
    // Case 2: 内存受限，按负载比例分配 replica
    totSlots ← |D| * m
    for each e in sortByLoadDescending(E^topT):
        n ← assignSlotsByLoad(e, totSlots, F)
        // 如 e 有 30% 总负载 → 分配 totSlots * 30% 个 slots
        P^e ← Distribute n replicas of expert e across
              nodes and devices, prioritizing nodes with
              more available slots
        P' ← P' ∪ P^e

// Calibration (optional, on critical path):
// 在 MoE gate 输出实际 token assignment 后，重新运行 Algorithm 1
// 若 calibrated placement 的延迟收益 > 额外通信开销 → 接受
return P'
```

overlap degree t 的计算：`t = T_non-MoE * bw / expert_size`，其中 T_non-MoE 是 attention layer 的计算延迟（可 profiled 或实时获取），bw 是 inter-node bandwidth（异构网络）/ uniform inter-device bandwidth（同构网络），expert_size 是单个 expert 参数的字节数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Sparse materialization 在每次 iteration 的 forward pass 中执行（与 attention computation 重叠），不在 critical path 上。
- Calibration 阶段可选地在 critical path 上额外执行（MoE gate 输出后、token dispatch 前），仅当追加的 SparseAllGather 带来的 load balance 改善超过其通信延迟时才执行。
- 结合 re-materialization：forward 后释放物化参数，backward 前重新执行 sparse materialization（通过 spAG 重新物化）。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
