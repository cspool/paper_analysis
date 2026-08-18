## Asynchronous Back-Streaming（异步背流协议 / AXLE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asynchronous Back-Streaming 是 AXLE 提出的第三种 CCM 卸载协议：让 CXL 设备（而非主机）触发反向数据流——CCM 的 DMA executor 把部分结果经 CXL.io DMA 提前"背流"到主机本地内存，使数据搬运与 CCM 执行、主机执行三方连续重叠；控制面沿用 CXL.mem store（kernel 启动 + 流控消息），保留 BS 的低协议开销。灵感来自 CXL.mem 的 back-invalidation snoop（设备发起一致性消息），但 back-invalidation 不携带 payload，故以 bus-master DMA + 事务层之上的自制流控实现，不改 CXL 协议。四项配套机制：本地轻量轮询（轮询点移到本地 metadata ring 尾指针）、metadata/payload 双 ring buffer + ready pool、OoO 乱序流式（gap-aware 消费）、内存正确性保障（栅栏排序 + 保守 head/tail + cache-bypass）。别名/相关：KAI（早期版本名）、异步背流、back-streaming。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PageRank 迭代的背流时序（AXLE Fig.9）：主机 CXL.mem store 启动 kernel（不阻塞）→ µthreads 计算并写结果 → DMA executor 打包 payload+metadata、达 SF 即 CXL.io DMA 反流 → 主机轮询本地 metadata 尾指针（间隔 50ns/500ns/5µs）→ 搬 metadata 入 ready pool → 调度器选就绪任务读本地 payload 执行下游计算 → CXL.mem 回传 head 索引流控。三方重叠使最长组件隐藏其余：PageRank 端到端较 RP -50.14%、较 BS -48.88%；CCM idle 平均 -13.99×（vs RP）/-14.53×（vs BS）、host idle 平均 -3.93×、host core stall 最高 -6×。中断通知变体（50µs 中断延迟）在细粒度任务下反而劣于基线（KNN 214.64%），证明"本地轮询"是通知的关键。LLM 案例收益小（attention 输出 [1, hidden_size] 稀疏、主机任务少，重叠机会有限）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：三层软件栈——主机内核态驱动（预 pin DMA 区域、scatter-gather 描述符影子化、cache-bypass）+ 主机用户态（流控消息等协议行为）+ 设备固件（处理卸载请求、监控结果、DMA executor 编程）。参数：SF（32B 默认，SF2–SF32 在数据搬运重型负载下 -7% runtime，SF_50%+ 退化）、DMA slot 容量（默认 50000，12.5% 仍稳）、OoO 开关（RR 调度下关闭 +1.38~1.74× runtime）。使用方式：通用 CCM 部分卸载；多租户扩展需动态 SF/轮询间隔与显式完成通知。边界：LLM 稀疏依赖 + OoO + 极小缓冲容量会死锁（需充足缓冲或顺序流式）。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
