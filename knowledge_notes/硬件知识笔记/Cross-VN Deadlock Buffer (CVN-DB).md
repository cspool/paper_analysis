## Cross-VN Deadlock Buffer (CVN-DB)

术语是什么？

CVN-DB (Cross-Virtual Network Deadlock Buffer) 是 DFBM 的共享 deadlock buffer 模块，位于 chiplet-to-interposer 输出路径上，用于在输出端拥塞时临时存储可能造成堵塞的包，避免 backpressure 传播回 chiplet 内部 NoC 形成跨域死锁环。与传统方案为每个 virtual network (VN) 分配独立 deadlock buffer（dedicated per-VN buffer）不同，CVN-DB 让多个 VN 共享同一个 buffer pool，通过 VN 依赖优先级（Response > Forward-Request > Request）控制入队/出队顺序，从而将面积开销从 ~5% 降低到 ~2.5%。

从硬件架构角度拆解术语：

CVN-DB 的优先级仲裁逻辑基于 VN 依赖链的 topological order：在 cache coherence protocol 中，Request (VN0) 可能触发 Forward-Request (VN1)，Forward-Request 可能触发 Response (VN2)。依赖关系为 Request → Forward-Request → Response，即 Response 是依赖链的末端（sink），Forward-Request 是中间节点，Request 是依赖链的起点（source）。CVN-DB 的入队策略：优先保证高优先级（靠近依赖链末端）的包有空间——Response (VN2) 总是可入队，Forward-Request (VN1) 在 buffer 剩余空间足够时入队，Request (VN0) 仅在保留空间充足时入队。出队策略：高优先级包先被 drain（Response > Forward-Request > Request），确保依赖链末端的包优先离开，释放依赖。Sensitivity study 表明：CVN-DB 容量需与 VC 数匹配——VC=4 时容量变化影响较小（因为 VC 本身提供足够弹性），VC=2 时 buffer 容量不足会明显限制延迟和吞吐。

术语一般如何实现？如何使用？

CVN-DB 以 shared SRAM buffer + priority queue controller (Verilog RTL) 实现。Buffer 组织方式：单一物理 SRAM 被多个 VN 逻辑分区共享（而非 per-VN 独立 SRAM），通过 head/tail pointer per priority level 管理。入队时：controller 检查当前 occupancy 和入队包的 VN → 根据优先级规则（Response always, Forward-Request if >threshold, Request if >higher_threshold）决定是否允许入队 → 若允许则写入 buffer 并更新对应 priority level 的 tail pointer。出队时：priority arbiter 轮询各 priority level 的 head pointer → 最高非空优先级先出队 → 读取 buffer → 更新 head pointer 和 occupancy counter。关键 trade-off：shared buffer 相比 dedicated per-VN buffer 有边际性能下降（极端拥塞下高优先级包可能被低优先级包占用 space），但面积节省约一半（~5% → ~2.5%），且开销位于主动中介层侧（成本更低）。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

