## Layer-Level Decode Batching（层级解码批处理）

术语是什么？通过联网搜索让回答具体和精准。
Layer-Level Decode Batching是Laser decode阶段的差异化批处理机制，通过控制每个请求每iteration执行多少transformer layer来差异化处理不同SLO的decode请求。核心参数为每个请求的execution plan (L, O)：L表示每iteration执行的层数，O表示调度offset（延迟多少iteration后重新执行）。strict请求（如50ms TBT）每iteration执行全部N层保证低延迟，relaxed请求（如200ms TBT）每iteration只执行N/2或N/4层，多轮完成完整forward pass，释放batch slot给更多relaxed请求。这利用了relaxed请求对更高latency的容忍性（可容纳7×更多并发请求）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Layer-level decode batching的execution plan构造流程（Planner的ExecPlan算法核心）：
```
// ExecPlan(N, R, L, O, R*): 构造decode execution plan
// N: total layers, R: existing requests with L/O configs,
// R*: newly arrived requests

1. 新请求初始化: for r in R*: L_r = N, O_r = 0  // 全层执行无延迟
2. Iter = LatencyEstimate(L, O)  // 用modular latency model估计per-iteration延迟
3. T_g = min_{r in R} SLO_TBT_r  // 最严格TBT作为target

4. while True:
     if Iter > T_g:  // 延迟超标 → 减少relaxed请求的层数
         r_opt = argmax_{L_r > ceil(N*T_g/SLO_TBT_r)} SLO_TBT_r  // 选最relaxed请求
         L_{r_opt} = ceil(N * T_g / SLO_TBT_{r_opt})  // 降到SLO允许的最小值
         O_{r_opt} = argmin_{o in [0, M]} T(P[L_{r_opt}, o])  // 选offset平衡同组负载
         Iter = LatencyEstimate(L, O)
         if Iter < T_g: break
     else:  // 延迟低于target → 尝试恢复strict请求全层
         r_opt = argmin_{L_r < N} SLO_TBT_r  // 选最strict请求
         L_{r_opt} = N; O_{r_opt} = 0  // 恢复全层执行
         Iter = LatencyEstimate(L, O)
         if Iter > T_g: restore configuration; break

5. return L, O, R, Iter, T_g
```

执行语义：request r在iteration i执行layer j当且仅当 `(i - ceil(j/L_r) - O_r) % ceil(N/L_r) == 0`。例如，L=N（全层执行）时每个iteration所有层都执行；L=N/2时每2个iteration执行一轮完整前向，每个iteration只执行一半层，两个L=N/2但O不同的请求可以交错执行互补层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Laser中，decode Planner仅在critical events触发plan更新（request arrival/departure，或per-iteration latency接近最严格SLO），更新开销<3.8%且与model execution overlap。Planner模拟未来若干iteration取最大predicted latency作为plan的保守估计。更新采用贪心策略：优先最大化每次变更的影响幅度（先调整最relaxed请求），最小化总更新次数和因offset变更造成的request reordering。Laser实测layer-level batching降低TBT violation rate >6.7%，叠加group-based assignment后额外降低10.5%；TPOT violation rate降低最高11.8%。

涉及论文标题：
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

---
