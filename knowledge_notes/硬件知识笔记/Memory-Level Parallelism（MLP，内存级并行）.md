## Memory-Level Parallelism（MLP，内存级并行）

术语解释
处理器能同时在途（outstanding）的内存访问数量；MLP 越高，越能用重叠隐藏访存延迟。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
与指令级并行（ILP）相对，MLP 度量的是"内存流水线"而非"计算流水线"的利用率：一次 L2/DRAM miss 有数百周期延迟，单线程靠多个独立 miss 同时飞行来摊销。OoO 核的 MLP 受制于一系列硬件资源：LSQ（跟踪在途 load/store）、ROB/IQ 窗口（能看多远）、MSHR（每个 miss 占一项）、以及预取器的覆盖能力；哪一项先耗尽，MLP 就封顶。论文在 ICA 分析中展示了 MLP 对加速器的决定性影响：完美 ICA（零计算时间）下 SDDMM 的 memory-bound 周期仍占 62%+——计算加速了，但核的通用访存硬件喂不饱加速器，瓶颈从"算"转移到"取"。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文的 MLP 供给链对比：
- ICA 路径：核 load 指令 → LSQ（几十项级）→ L1 → L2 MSHR（论文放大到 128）→ DRAM；在途请求数受 LSQ/ROB 限制，prefetcher 失效时 MLP 骤降。
- ATX NCA 路径：一条 ATX 指令编码整个任务的所有访存 → UTE 的 32 个 Stream Units 各自生成地址 → 128 项 LDQ 挂起请求 → L2；在途请求数由 LDQ 深度决定，与核 ROB/LSQ 无关，因此 MLP 可远超核路径。论文还把"预测任务"（task prefetching）叠加其上进一步提前取数。roofline 图（图 16）给出佐证：ATX NCA 把 ridge point 大幅右移，使原本 compute-bound 的 SDDMM 全部落入 memory-bound 区。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现手段：深 MSHR、大 LSQ/ROB、runahead 执行、MLP-aware 指令窗口缩放、预取器（stride/stream/指针）。论文的实现要点：以"硬件模块（UTE）专用 LDQ/Stream Units"替代"核通用队列"承担加速器 MLP；以任务级 stride 预测（Predicted Prefetching）替代访存级预测。适用判断：memory-bound 且访存可并行化的负载最受益；纯 compute-bound 负载无需高 MLP。

涉及论文标题：
- ATX: Accelerator Task Extensions
