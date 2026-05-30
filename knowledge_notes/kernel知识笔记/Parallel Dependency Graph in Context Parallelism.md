## Parallel Dependency Graph in Context Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Parallel Dependency Graph（并行依赖图）是UltraAttn中描述分布式attention的computation和communication kernel依赖关系的DAG。在node/device-level tiling后，每GPU的computation kernel（FlashAttn）、send kernel（NCCL send）、recv kernel（NCCL recv）构成DAG节点，数据流依赖构成DAG边（recv→compute→send）。图结构由tiling结果决定，后续用于kernel-level tiling（贪心图变换）和ILP runtime scheduling。

从kernel调度角度拆解：
```
# 三种节点类型
comp_nodes = [A_{r,c} | allocated to this GPU]  # 矩形
recv_nodes = [需要的Q_r, KV_c from remote]       # 椭圆
send_nodes = [本地Q_r, KV_c to remote]           # 菱形
# 三种substitution类型
# 1.Comp batching: A0+A1→A0_fused (FlashAttn合并)
# 2.P2P comm batching: 同(src,dst)的send/recv合并
# 3.Collective batching: P2P→all-to-all
# Greedy selection:
for each transformation sorted by gain desc:
    if applicable: apply and keep if ILP_eval improves
```
**Annotations**: 节点数受pattern密度和P影响。Transformation gain = fused kernel time - sum of individual kernel times。与FlexFlow BFS-based scheduling不同，UltraAttn的ILP runtime找理论最优执行顺序。

术语一般如何实现？如何使用？依赖准确kernel profiling（FlashAttn各shape + NCCL各message size的$D_v$）。通信contention通过按共享带宽分组kernel到不同CUDA stream避免。最终DAG编译为CUDA graph消除CPU launch overhead。支持非对称、不规则workload（block sparse attention），与FlexFlow的symmetric SOAP搜索空间形成对比。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
