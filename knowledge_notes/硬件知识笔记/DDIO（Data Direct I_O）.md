## DDIO（Data Direct I/O）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Intel Data Direct I/O 是自 Xeon E5 起的硬件特性：I/O 设备（RNIC/NVMe 等）的 DMA 写入直接进入处理器 LLC 而非主存，写操作在数据到达缓存即视为完成，缓存行被淘汰时再后台写回 DRAM。效果是 CPU 后续读取命中缓存——论文引用的数据：LLC 访问约 29 CPU cycles、主存访问约 60 ns，CPU 访问远快于 RNIC 经 PCIe 的 ~1 µs。论文用它论证"把原子卸载到 CPU"的可行性基础：RNIC 直写 LLC（DDIO）缩短 CPU 数据访问路径，加上 CPU cache-coherence 对共享内存原子比 RNIC 锁定表更高效（θ=0.99 时 HERD 反超 RNIC-Only 1.4×）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 入站写事务流程（Web 证据：Intel DDIO 性能监控文档）：设备 DMA 写经 Integrated I/O → M2IOSF 拆成 64B cache line 粒度 → CHA（Caching/Home Agent）按 MESIF 协议获取该行所有权（ITOM，必要时 snoop 失效其他副本）→ 数据写合并进 LLC（默认写分配；LLC miss 且不分配时直接旁路）。论文场景：Fusa-RPC 把 CAS 请求写入 server 主存请求缓冲 → DDIO 使该数据落入 LLC → server CPU 线程以近缓存延迟读取并执行原子 → 结果同样经缓存/主存返回。DDIO 是 Fusa"CPU 软件路径快"的硬件前提之一。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- BIOS 默认开启、软件透明；I/O 写分配受 Intel CAT 静态限制只占 LLC 一部分，防 I/O 突发抖动整体缓存。代价：LLC 是共享资源，大队列深度存储负载下 DDIO 可能得不偿失；且共享 LLC 引入安全面（NetCAT 侧信道攻击，Intel 建议不可信网络下禁用 DDIO/RDMA）。论文测试台（Intel Xeon Silver 4314）默认启用 DDIO；论文引 [19][20][28] 作为 LLC/DDIO 行为依据。

涉及论文标题：
- Breaking Barriers in Atomic Scaling: A Hardware–Software-Collaborated Framework to Deconstruct RDMA Atomic
