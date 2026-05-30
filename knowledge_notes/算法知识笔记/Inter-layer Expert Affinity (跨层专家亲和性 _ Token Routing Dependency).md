## Inter-layer Expert Affinity (跨层专家亲和性 / Token Routing Dependency)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Inter-layer Expert Affinity（跨层专家亲和性）是 MoETuner 发现并利用的 MoE 模型关键特性：token 在相邻 MoE 层之间的路由决策不是独立的，而是遵循可预测的依赖模式。当 token 在层 l 被路由到 expert e_1 时，该 token 在层 l+1 更倾向于被路由到特定的少数几个 expert（而非均匀分布到所有 E 个 expert）。度量指标 R_{e_1,e_2,l}：层 l→l+1 间从 expert e_1 路由到 expert e_2 的 token 数量。Mixtral-8x7B 分析表明此路由模式跨 batch 高度一致，可使用数据集采样子集准确近似整体行为。

从算法pipeline角度拆解术语：
MoE 模型逐层推理时，对每个 token 追踪跨层路由路径：
```
for l in range(L-1):
    gate_l = Softmax(h_l @ W_gate[l])       # [E] router logits
    top2_l = TopK(gate_l, 2)                # 层 l 选中的 2 个 expert
    gate_l1 = Softmax(h_{l+1} @ W_gate[l+1])
    top2_l1 = TopK(gate_l1, 2)              # 层 l+1 选中的 2 个 expert
    # 统计跨层 expert 对
    for e_src in top2_l:
        for e_dst in top2_l1:
            R[e_src][e_dst][l] += 1         # 累加路由计数
```
MoETuner 利用此属性：在 ILP 2 中，若 R_{e_1,e_2,l} 很大（expert e_1 和 e_2 频繁被同一 token 连续层激活），则将其放置在同一 GPU → 消除该 token 在层 l→l+1 间的跨 GPU 通信。这是一种数据驱动的编译优化：将运行时 token 路由模式转化为离线 expert placement 决策。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 发现方法：在任务数据集采样子集上运行推理，收集逐 token 跨层路由路径，构建频率表。MoETuner 证明采样子集足以捕获整体路由模式。
- 应用：(1) Expert Placement——高亲和性 expert 对放同一 GPU。(2) Expert Prefetching——预测下层可能激活的 expert 提前加载。(3) Network Scheduling——预分配高通信量 GPU pair 带宽。
- 相关研究：ExFlow (IPDPS 2024) 同样利用 inter-layer expert affinity 做 locality-aware expert placement（graph-based, 非 ILP）。
- 局限：依赖特定任务数据集，切换任务需重新 profiling。

涉及论文标题：
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
