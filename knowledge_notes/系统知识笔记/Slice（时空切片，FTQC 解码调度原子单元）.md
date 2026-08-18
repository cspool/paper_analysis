## Slice（时空切片，FTQC 解码调度原子单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Slice 是 Triage 论文引入的解码调度原子单元：S(t,p)=一个 d×d 逻辑 patch 在 t 时刻（一个 d 轮 syndrome 测量周期）内产生的全部 syndrome 数据块，t 为时间索引、p 为空间位置。逻辑链：解码器处理大块 syndrome 有超线性复杂度惩罚（解码整块不如分别解码其组成部分），所以把 lattice surgery 计算体积按"空间 patch × 时间轮"切成细粒度 slice；slice 之间以无向约束图 G=(V,E) 组织，边=互斥约束（两 slice 不能并发解码），最多 6 邻居：2 个时间邻居（t−1、t+1 同 patch）与 4 个空间邻居（t 时刻相邻 patch，由 multi-qubit lattice surgery 引入）。每个 slice 带属性：解码状态（UNGENERATED→PENDING→OCCUPIED→ASSIGNED→COMPLETED）、deadline（到最近关键同步点的层数，无则为∞）、可能的因果锥引用。这比"逻辑量子比特级"的调度抽象更细，能同时利用时间与空间并行，并避免解码大合并区。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Slice 是调度器决策的输入粒度，驱动整个在线调度循环：
```
# Triage 的 slice 生命周期（每个 syndrome 到达/解码完成事件）
1) 硬件产生一层 syndrome → 相关 slice 变 PENDING（进入 Timeline）
2) 调度器选 ≤M_available 个无冲突（独立集）PENDING slice：
   稳态：P(V)=w_u·(1/Deadline)+w_c·(1/(Degree+1)) 加权启发式
   紧急：按预计算计划派发（预测性因果锥着色）
3) 被选 slice → ASSIGNED，解码器占用（其邻居被标 OCCUPIED 阻塞）
4) 解码完成 → COMPLETED，释放解码器 → 重触发调度
5) 关键操作前检查因果锥是否全 COMPLETED，否则插入 idle layer
```
Timeline 中每个 unit 存 (layer t, 坐标(r,c), 操作标签, 6-bit 邻居 mask, deadline, causal cone 引用)；调度时间 τ 上 syndrome 到达每 1 cycle 一步、解码完成事件在 τ_finish=τ_start+T_dec。slice 的 degree（未解析邻居数）同时决定解码延迟（窗口缓冲大小）与调度代价（Cost-Efficiency=1/(Degree+1)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译期由静态分析器从 LLI 构建 Timeline（离线）；在线由离散事件模拟器维护 PENDING/COMPLETED 集。使用场景：M-for-N 解码器池调度的最小并行单元——把 lattice surgery 合并区切成空间可并行 slice、把时间流切成时间可并行 slice，同时避免超线性复杂度惩罚。与通用 KV-cache 的 VA slice（ConServe/vAttention）无关，是 FTQC 特有的时空调度单元。论文未开源实现，Python 3.9 仿真框架复现。

涉及论文标题：
- Triage An Adaptive Parallel Window Decoding Scheduler for Real-time Fault-Tolerant Quantum Computation
