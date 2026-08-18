## HCA Atomic 与 RNIC 内部锁定表（lock slot、PU、PCIe RMW）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HCA Atomic 是 RNIC 内部保证 RDMA 原子（CAS/FAA）正确性的硬件原子模型：每个 8 字节对齐的原子操作按目标地址哈希映射到内部锁定表的一个槽（Mellanox RNIC 为 512 槽，来自 [80] 的逆向工程结论），映射到同一槽的并发原子被强制串行；被分配的处理单元（PU）以 PCIe Read（取旧值）+ PCIe Write（回写新值）完成一次 RMW。问题：倾斜访问下热地址聚到少数槽 → 槽级争用 → 串行化，论文实测原子吞吐最高下降 94.5%。Web 证据：NAM-DB/SIGMOD 相关论文同样确认 RNIC 原子在锁槽内串行、latch stride 越大 lock-table 碰撞越重（core.ac.uk 与 TUD SIGMOD'23 PDF）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 服务端执行流程：RDMA_CAS 到达 RNIC → 地址哈希得槽号 → 槽锁仲裁（同槽请求排队，①）→ 分配的 PU 发 PCIe Read 取当前值（②）→ 与 compare 值比较：不等则只回旧值；相等则 PCIe Write 写新值（③）→ 生成 CQE。Fusa 的对应改造：把每个槽按地址额外 g 位细分为 group（默认 8,192 组，即槽粒度再分 16 组），把"槽级硬调度"细化为"组级软调度"——槽仍保证原子性，但组决定该请求走 RNIC 还是卸载到 CPU。
- 世代差异：CX-5 与 CX-6 锁定表行为不同（CX-5-PA 同时用 PCIe Atomic 与锁定表，stride>512B 时瓶颈回到锁定表、性能最差）；论文以 stride 8B–8,192B × 128 线程实验刻画槽映射（stride>128B 时 CX-5 吞吐开始下降）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 槽数与哈希方式是厂商私有实现，论文靠 stride 实验 + 逆向工程（[80]）确认 512 槽；其他厂商 RNIC 的槽数可用同样方法探测。增强原子（masked CAS 等，见 Mellanox Advanced Transport 文档）只走 HCA 路径，PCIe Atomic 不支持。使用上：无争用/低倾斜负载下锁定表是最高吞吐路径（CX-6 stride 8B 达 42.3 Mops/s），这正是 Fusa 保留"RNIC 路径"的原因。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
