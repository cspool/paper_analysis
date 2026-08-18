## 因果锥（Causal Cone）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
因果锥（causal cone）是 Triage 为关键操作（T 门 teleportation 后的 Clifford 校正）定义的历史依赖闭包：为更新 Pauli frame 而必须解码完的、该关键 slice 的所有历史 slice 集合——即通过一串 multi-qubit 操作与目标相关的所有逻辑量子比特上、尚未解码的历史 slice 的传递闭包。逻辑链：lattice surgery 合并使错误跨 patch 相关，T 门同步点要求 E_acc 被物理纠正，而 E_acc 的推断依赖所有参与过合并的历史 syndrome；若算法高度纠缠且非 Clifford 门间隔长，因果锥可长成很大的时空体积——brute-force 在最后一刻并行解析这种积压需要与积压规模成比例的解码器数，这正是"资源可扩展性危机"：解码器需求高度不均匀，T 门之前需求尖峰，静态最坏情况配置解码器在架构上不可行。Triage 用按需计算（lazy）实现因果锥：调度器需要时从关键 slice 的时空前驱反向 BFS（只扩同层空间邻居 + t−1 时间前驱，COMPLETED 剪枝），结果存有界 LRU 缓存，后续同关键操作查询 O(1)。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
因果锥是 Triage 紧急模式的范围与目标：
```
# 紧急模式启动流程
PENDING slice 的 deadline ≤ τ_emergency(4) → Triage Trigger 触发
C = BackwardBFS(关键 slice)     # 时空前驱闭包，COMPLETED 剪枝，LRU 缓存
if |C| > ScopeCap(100): 回退稳态模式          # 防规划延迟尖峰
else: 计划 P = PredictiveColoring(C)          # Algorithm 1，O(n log n)
      按 P 派发，目标：同步点前把 C 全部解码
```
因果锥规模（scope）直接决定紧急模式计算开销：scope 增长时计划时间按 O(n log n) 增长（拟合 y=a·n·log n, a=0.01513, R²=0.8056）。ScopeCap 防止应用末段累积的巨型因果锥进入关键路径（无 ScopeCap 时 delay ratio 0.06 即 backlog 失败）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BFS + 有界 LRU 缓存（论文明确），节点=slice，边=时空依赖；在线调度器在 Triage Trigger 判定时按需计算。使用场景：决定"紧急模式要预解码什么"与"同步是否满足"（每关键操作前检查因果锥是否解码完，否则插 idle 层）。本质是 FTQC 版的依赖闭包调度概念（类似任务依赖图中的传递闭包），但以 slice 为粒度、以 Pauli frame 同步为 deadline。论文未开源实现。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
