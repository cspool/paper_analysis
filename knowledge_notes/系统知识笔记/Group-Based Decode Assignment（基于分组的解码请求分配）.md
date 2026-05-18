## Group-Based Decode Assignment（基于分组的解码请求分配）

术语是什么？通过联网搜索让回答具体和精准。
Group-Based Decode Assignment是Laser Global Controller中将decode请求分配到decode instance的算法。核心思想：维护SLO-homogeneous instance groups（按TBT target将decode instance分组），新请求优先分配到SLO最匹配的group中TBT increment最小的instance。group设计减少同一instance内TBT需求差异度，提升layer-level decode batching的colocation效率。group大小按各SLO target的arrival rate与per-instance throughput比值动态调整。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Group-based decode assignment算法流程：
```
Function DecodeAssignment(G, r*):
  // G: SLO-homogeneous instance groups
  // r*: new decode request
  for d in all_decode_instances:
      I_d = LatencyIncrement(r*, d)  // 去中心化: 各instance本地评估
  sort G by |SLO_g - SLO_TBT_r*|  // 按SLO匹配度排序
  for g in G:
      I_opt, d_opt = min_{d in g} I_d, argmin_{d in g} I_d
      if I_opt != INF:  // 存在SLO-compliant的instance
          return d_opt
  // fallback: 跨组或best-effort
  assign r* to least-loaded instance

Function LatencyIncrement(r*, d):
  // 在decode instance d上本地评估接纳后的影响
  N, R, L, O = current state of instance d
  Iter = LatencyEstimate(L, O)  // 当前per-iteration延迟
  L*, O*, R*, Iter*, T_g* = ExecPlan(N, R, L, O, r*)  // 接纳r*后的新plan
  if Iter* > T_g* or Memory_Shortage:
      return INF  // 拒绝: 会导致SLO violation或OOM
  else:
      return Σ_{r in R*} Iter* * ceil(N/L_r*) - Σ_{r in R} Iter * ceil(N/L_r)
      // aggregated TBT increment across all requests
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Laser Global Controller中实现。两个关键设计：(1) 去中心化性能评估——每个decode instance本地运行ExecPlan评估接纳影响，仅向controller返回aggregated TBT increment，使dispatch overhead保持~10ms且不随instance数增长（vs 中央评估方案的线性增长）；(2) 动态group sizing——各group的instance数按 `arrival_rate / per_instance_throughput_at_max_SLO_compliant_batch` 计算，并根据workload变化实时调整（groups是虚拟的，可灵活resize）。Laser实测group-based assignment在layer-level batching基础上额外降低10.5% decode SLO violation rate。

涉及论文标题：
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

---
