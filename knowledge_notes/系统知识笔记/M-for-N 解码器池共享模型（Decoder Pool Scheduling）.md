## M-for-N 解码器池共享模型（Decoder Pool Scheduling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
M-for-N 共享资源模型是 FTQC 解码器部署的资源模型：M 个物理解码器服务 N 个逻辑量子比特（M<N）的持续 syndrome 流。逻辑链：两个极端都不可行——one-to-one（每逻辑比特一个专用解码器）代价高、利用率低、大规模架构不可行；one-for-all（常数解码器服务全机）每解码器延迟需求 O(1/N_lq)，非平凡算法不可能满足。唯一可行是共享资源池 + 在线调度器做动态分配：把"哪个解码器解哪个逻辑 patch"变成实时调度问题。两个关键场景：资源受限区（M≤N, τ_dec<τ_gen，快解码器稀缺）与计算受限区（M>N, τ_dec≥τ_gen，解码器可慢但多）。Triage 以 slice 为调度粒度把该模型细化：每决策点选一个独立集 V'⊆PENDING 满足 |V'|≤M_available，目标是极小化插入 idle layer 数（即极小化 LER）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
M-for-N 池调度的在线循环（Triage 形式化）：
```
给定：共享池 M 个解码器（各 r_dec 倍 syndrome 生成速度）、动态无向约束图 G=(V,E)
决策点：syndrome 到达 / 解码完成
任务：找分配 π: V' → {1..M}，V' 是 G 的独立集，|V'| ≤ M_available
目标：极小化同步失败导致的 idle 层总数（→ 极小化 LER）
# 执行：稳态启发式每事件选独立集；紧急模式按预测计划批量派发；回填填空隙
```
评估：扫描解码器数 M 与相对速度 τ_dec/τ_gen 画热力图（idle 层数）与最优调度器地图——Triage 定义资源受限下界前沿，SWIPER 在资源充裕上界区最优，极稀缺区（左下角黑色）所有调度器触发 backlog 终止。论文结论：在慢解码器区（τ_dec>τ_gen）通过调度并行窗口也能维持 FTQC。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文用 Python 3.9 离散事件模拟器复现（解码器延迟=pymatching 幂律模型）；baseline 含 serial sliding window、time-parallel window、SWIPER。使用场景：FTQC 经典控制层的资源分配——与 [28]（Maurya & Tannu 的逻辑量子比特级调度）相比，Triage 把抽象下放到 slice 级以利用时空并行。论文未开源实现。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
