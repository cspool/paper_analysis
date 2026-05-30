## Hierarchical Context-Tiling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Context-Tiling（分层上下文瓦片化）是UltraAttn提出的三层context parallelism优化框架：(1) Node-Level——将跨节点通信解耦，每个node视为集成设备，通过node间groupwise peer-to-peer最小化通信；(2) Device-Level——在node内通过ILP在$P \times P$ grid上分配attention blocks到GPU，在负载均衡约束下最小化peer-to-peer通信；(3) Kernel-Level——在parallel dependency graph (DAG)上通过贪心图变换（computation/communication kernel batching）自适应调整kernel粒度，平衡kernel overlap与单kernel device utilization。核心理念：沿Q和KV两个维度同时tile（curled-up形状$\sqrt{N} \times \sqrt{N}$），将通信projection从O(N)降至$O(\sqrt{N})$。

从kernel调度角度拆解，三层tiling的执行伪代码：
```
P = find_min_P(DLI_{P,CP} ≤ θ_{DLI})
# Node-Level ILP (CP_node=8)
ILP_solve(x_{r,c,g}, FB, CB, EB, CP=8, minimize MCV)
# Device-Level ILP (CP_device=8 per node)
for each node: ILP_solve(same formulation, CP=8)
# Build DAG: comp kernels + send kernels + recv kernels
G = build_dag(allocations)
# Kernel-Level: greedy DAG transform
candidates = [comp_batching, comm_batching, collective_batching]
candidates.sort(by=transformation_gain, descending=True)
for cand in candidates:
    if applicable: G_try = apply(cand, G)
    if ILP_runtime_eval(G_try) < ILP_runtime_eval(G):
        G = G_try
# ILP Runtime: group by bandwidth → per-stream ILP scheduling
streams = group_by_bandwidth(G.kernels)
for stream: ILP_solve(S_v, Order_{uv}, min End_Time)
```
**Annotations**: Communication volume weights: Q:KV:O = 1:2:1（per-token数据量比）。Cmap映射来自context remap决定哪个device持有哪个context chunk。Greedy transformation基于局部交互假设。

术语一般如何实现？如何使用？需要Gurobi ILP solver + FlashAttn profiling + NCCL profiling。Attention pattern以FB/CB/EB集合输入，自动完成三层tiling。ILP时间：strided (P=2) 0.07ms → causal (P=8) 3672ms。适用场景：long-context LLM training/inference with block sparse attention，特别是跨多节点（CP>8）。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
