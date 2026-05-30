## Stream Assignment Algorithm (Minimum Equivalent Graph + Bipartite Matching)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stream Assignment Algorithm 是 Nimble 自动将 DL 模型 DAG 中的算子分配到多个 CUDA stream 的算法，目标是实现最大逻辑并发度（maximum logical concurrency）且最小化跨 stream 同步次数（minimum number of synchronizations）。

算法基于图论，分四步：
1. **构建 Minimum Equivalent Graph (MEG)**：将原始 DAG 简化为最小等价图——去除冗余传递边（transitive edges）。如果 A→B→C 且 A→C 存在，则 A→C 是冗余边，去除后信息量不变但图结构简化。MEG 暴露了真正的直接依赖关系。
2. **构建 Bipartite Graph**：从 MEG 出发构建二分图。MEG 中的每条有向边 (u, v) 成为二分图中的节点；如果两条边 (u₁, v₁) 和 (u₂, v₂) 在 MEG 中不相交（无共同节点），则它们可以分配给不同 stream，在二分图中连接为一条边。
3. **Maximum Matching (Ford-Fulkerson)**：在二分图上运行最大匹配算法。每个 matching 代表一组可并行执行（无共同节点 → 无依赖冲突）的边。最大匹配的数量决定了最优 stream 数量——每个 matching 分配给一个 stream。
4. **Stream Assignment**：基于 maximum matching 结果，将 MEG 中的节点（算子）分配到对应 stream。同一条 stream 中算子按拓扑序排列，跨 stream 在 matching 边界插入 CUDA event 同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Algorithm: Stream Assignment for GPU Multi-Stream Execution
// Input: DAG G = (V, E) where V = operators, E = data dependencies
// Output: stream_assignment: map<v ∈ V, stream_id>

function assign_streams(G):
    // Step 1: Minimum Equivalent Graph (MEG)
    // Remove redundant transitive edges
    MEG = compute_closure(G)     // 计算传递闭包
    for each edge e = (u, v) in G:
        if exists path u → ... → v with length > 1 (excluding e):
            remove e from G       // e is a redundant transitive edge
    // G is now the MEG

    // Step 2: Construct Bipartite Graph
    // Bipartite nodes: each edge in MEG → 2 copies (left and right)
    left_nodes = {}; right_nodes = {}
    for each edge e = (u, v) in MEG:
        left_nodes.insert(L_e); right_nodes.insert(R_e)
    
    bipartite_edges = {}
    for each pair of edges e1 = (u1, v1), e2 = (u2, v2) in MEG:
        if e1 and e2 share no vertices:  // u1≠u2≠v1≠v2
            if e1 precedes e2 (no path from v2 to u1):
                bipartite_edges.insert(L_e1 → R_e2)

    // Step 3: Maximum Matching (Ford-Fulkerson)
    matching = max_matching(bipartite_graph)
    // matching = set of disjoint edges in bipartite graph
    // |matching| = number of parallelizable edge groups

    // Step 4: Assign streams
    num_streams = |matching|
    group edges by matching chain:
        // 每条 matching chain 中的 MEG edges → 同一 stream
        // 同一 matching 中的边共享无冲突 → 可放入同一 stream
    
    for each matching chain:
        stream = new CUDA stream
        for each edge e in chain (topological order):
            assign(e.source, stream)
            assign(e.target, stream)
    
    // Insert CUDA events at cross-stream dependency points
    for each e = (u, v) where stream(u) ≠ stream(v):
        insert_cuda_event(e, stream(u), stream(v))

    return stream_assignment
```

具体例子（以图 3 的简化 DAG 为例）：
```
// DAG: A → B → C → D → E
//       A ──────────→ E  (transitive edge)

// 为什么需要 MEG 而非仅用 max-flow:
// max_flow(A→E, DAG) = 1 (每条 A→E 路径都包含 edge (A,B))
// → 暗示"最大并行度 = 1" — 误导性结论!
// MEG 去除 A→E (冗余传递边) 后:
// MEG: A → B → C → D → E
// → 链路中 B 在 A 完成后才能开始，C 在 B 完成后才能开始
// → 正确反映了真正的并行度限制
//
// 论文强调: "the maximum flow of graph is trivially 1, 
// and does not give useful information for the stream assignment"
// MEG + bipartite matching 提供更精确的并行度分析
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Nimble 在 Graph Rewriter 组件中实现了该算法。实现基于 PyTorch TorchScript graph 的 IR 分析：首先从 `torch.jit.trace` 的 traced graph 提取 operator-level DAG，然后在 CPU 端执行 stream assignment（AoT preparation 阶段），最后在 CUDA Graph capture 时按分配在对应 stream 上执行各算子。

理论保证：论文证明该算法实现了 maximum logical concurrency（DAG 中可并行执行的最大算子数）且 minimum number of synchronizations（跨 stream 的 CUDA event 同步点最少）。

实际效果：NASNet-A mobile model 的 maximum logical concurrency 达到 15（最多 15 个可并行算子），multi-stream 自身贡献 up to 1.88× speedup（在 AoT scheduling 之上的额外加速）。

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---
