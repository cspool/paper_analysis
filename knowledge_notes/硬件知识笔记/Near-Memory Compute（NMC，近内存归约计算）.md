## Near-Memory Compute（NMC，近内存归约计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Near-Memory Compute（NMC）是 MTIA 300 Message Engine 内的归约计算块，位于芯片边缘紧邻 HBM、cache 与 I/O，是"近内存计算"在归约类操作上的具体实现。规格：128 B/cycle 用于归约或 DMA，若所有 ME 的归约与 DMA 同时活跃则降为 96 B/cycle；16 个 ME 合计最高 2.8 TB/s 归约带宽，是 I/O 带宽（1.2 TB/s）的 2 倍以上。NMC 用于所有基于归约的 collective：Reduce、AllReduce、ReduceScatter。设计动机：把归约从 PE 网格卸载（PE 面积做归约不划算）+ 让归约在数据所在处（HBM/cache 旁）就地执行，减少数据搬移与 NoC 拥塞。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
NMC 的运转流程（ReduceScatter 阶段）：WQE 的 REDUCE 操作在 ME 的 NMC 执行 S=A+B（S 可与 A 或 B 重叠，或可选做内存拷贝充当 DMA 引擎）→ NMC 就近从 HBM/cache 读入参与归约的分片、累加后写回/转发 → 其靠近 cache/HBM 使 AllReduce/ReduceScatter 的关键操作不必占用 PE 网格与主 NoC。对比 PNM/Processing-Near-Memory（通用近内存计算，见 CENT/NeuPIMs 条目）：NMC 是"加速器内部、面向归约原语"的专用近内存块，而非 PIM 的通用计算；它强调"归约吞吐 > I/O 带宽"（2.8 vs 1.2 TB/s）以保证通信不被归约能力限制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每 ME 一个 NMC，128 B/cycle 数据宽，归约与 DMA 共用（全并发降速）；与 ME 的 CPU-M/NIC interface 协同；流控由 WQE 的 WAIT/SET 比较器协调（如 wait 地址 0xabcdef > 10）。使用场景：AllReduce/ReduceScatter 的梯度归约、AllGather 的数据搬移（REDUCE WQE 可选做内存拷贝）。信息缺口：论文未给出 NMC 的加法树/浮点格式细节与面积功耗。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
