## DMA Executor（AXLE 的 DMA 执行器 / Bus-Master DMA 引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DMA executor 是 AXLE 在 CCM 设备侧新增的数据面组件：作为 CXL Type 3 设备上的 bus-master DMA 引擎，监控 µthread 结果数据的生成，把连续结果打包成 payload（DMA slot 大小 = 主机 ring buffer slot 大小，默认 32B），为每个 payload 生成一条 metadata，当待发 payload 总量达到 Streaming Factor（SF）时经 CXL.io（PCIe）posted write 把 payload+metadata 写到主机本地 DMA 区域。这使设备（而非主机）触发结果搬运，是异步背流协议的核心硬件组件，设备固件可编程（经 OS 影子化的 DMA 区域描述符指定源/目的地址）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：µthreads 写结果到设备内存 → DMA executor 监测（结果连续达到 32B 即形成一个 payload、配一条 metadata）→ 待发总量 ≥ SF（32B 默认）触发一次 DMA 批（payload 写主机 payload ring、metadata 写主机 metadata ring，DMA 准备延迟 500ns/请求）→ 主机本地轮询 metadata 尾指针感知到达。正确性设计：内存栅栏保证 payload 完全写入先于 metadata tail 更新（防部分读）；主机经 CXL.mem 回传 head 索引作流控（防固定大小 ring 溢出）；CCM 本地维护保守 head/tail，陈旧流控消息仍安全（tail 不越过过时 head 即可继续流式）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：总线主设备 DMA 引擎（ASIC/FPGA 均可），描述符由主机 OS 预配置（预 pin 的 scatter-gather 区域、cache-bypass）；在 AXLE 模拟器实现中 DMA 准备按 500ns 单程控制面延迟建模，数据写入按内存操作显式模拟。使用方式：设备发起的结果流、反向数据搬运（与 RDMA 硬件信用管理的对照：背流非原生 CXL 协议，需在事务层之上自行做流控）。参数敏感性：DMA slot 容量（默认 50000）降到 12.5% 多数负载性能不变，但 LLM 稀疏依赖 + OoO + 极小容量会死锁。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
