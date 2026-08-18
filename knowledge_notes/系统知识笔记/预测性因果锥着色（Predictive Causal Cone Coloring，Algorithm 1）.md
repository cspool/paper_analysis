## 预测性因果锥着色（Predictive Causal Cone Coloring，Algorithm 1）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
预测性因果锥着色是 Triage 紧急模式的核心算法：把因果锥内所有 PENDING slice 的并行解码计划一次性算出来。逻辑链：一旦进入紧急模式，因果锥内所有 slice 共享同一紧迫性（都必须在同步点前解码完），吞吐瓶颈只剩计算代价 → 用 MDF（最小 degree 优先）贪心；算法跑一个离散事件模拟：初始化优先级队列（只放因果锥内 PENDING slice，输入最小化），主循环把模拟时钟推到下一事件，贪心选独立集（冲突无关的可用 slice），把 (t_sim, s) 记入计划 P，更新邻居的 t_start 与 degree，直到队列空。复杂度 O(n log n)（n=因果锥内 slice 数：初始化 O(n log n)、每 slice 出队一次、邻居更新至多 6n 次每次 O(log n)、排序 O(n log n)）。计划缓存后在线只做轻量派发表执行，因此只有无法被解码隐藏的规划/派发/互联延迟进入关键路径。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Algorithm 1 的输入→输出流程（论文伪代码）：
```
Input: 因果锥 slice 集 C, 当前时间 t_now, 解码器模型 D_model
Output: 紧急计划 P
Q ← ∅
for s ∈ C: s.t_start ← max(t_now, s.t_syndrome_ready); Q.push(s)   # 按 t_start 优先
while Q 非空:
    t_sim ← NextEvent(Q, D_model)
    R ← {s ∈ Q : s.t_start ≤ t_sim}         # 就绪集
    R.sort(by degree)                        # MDF
    N_free ← D_model.num_free(t_sim)
    D_dispatch ← SelectConflictFree(R, N_free)
    for s ∈ D_dispatch:
        P.add((t_sim, s))
        t_fin ← t_sim + CalculateDuration(s.degree)   # 幂律延迟
        for n ∈ Q 邻居 of s: n.t_start ← max(n.t_start, t_fin); n.degree −= 1; 更新 Q 位置
    R 中未派发 slice 重入 Q
return P
```
在线执行：调度器转成简单执行器，按计划的开始时间派发；机会回填用 M_peak（计划峰值解码器数）之外的空闲资源。与启发式"逐步决策"的本质区别：一次算完整因果锥的全局并行计划，保证同步点前完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：优先级队列 + 离散事件模拟 + 独立集贪心（论文明确，Python 原型）；复杂度 O(n log n)，ScopeCap=100 防巨型锥。使用场景：紧急模式下对即将同步的关键操作做最大并行度预解码；论文用 Delay Ratio（0-0.20）灵敏度模拟验证：无 ScopeCap 时 delay ratio 0.06 即 backlog 失败，有 ScopeCap 全程鲁棒。论文未开源实现。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
