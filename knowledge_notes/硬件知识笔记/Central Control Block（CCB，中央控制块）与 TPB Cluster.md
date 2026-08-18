## Central Control Block（CCB，中央控制块）与 TPB Cluster

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CCB 是 M100 NPU 的控制中心：4 核 SiFive X280 RISC-V CPU，每核配对自定义 vector engine，经 Instruction Chain Bus（ICB）派发 TPB 指令，支持最多 4 个并发推理任务；含 32MB 片上 SRAM（4×8MB bank、4KB 交织高带宽并行访问）、2 个 DDR↔CCB-SRAM DMA（可经 Data Ring Bus 直接向 TPB 广播权重，256GB/s 匹配 DDR 读带宽）、barrier 同步与中断生成，全部经 Arteris FlexNoC 互连。TPB Cluster 是 4 个 TPB 的中间层次：共享指令缓冲、ICB/DRB 节点与 1 个 SiFive X280 RISC-V Vector CPU（提升计算密度、共享面积），cluster 内 NoC 连接 4 TPB 与 CPU 内存口，经主/从端口接 NPU 级 Mesh Bus；近距离低延迟通信适合小规模 TPB 任务。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：主机调度 CPU 经 AXI slave 接口下发命令/检查状态 → CCB RISC-V 固件解析任务 → 经 ICB（daisy-chain，64 bits/cycle）以 destination mask 广播 TPB 指令到目标 cluster/TPB → cluster 指令队列按数据就绪与同步条件派发；CCB DMA 从 DDR 读权重经 DRB 广播到 TPB；barrier 让多 TPB 组在全局同步点对齐；任务完成后中断通知。TPB 需要 CPU 协助时经 CSU 触发 cluster CPU 中断服务例程（最多 4 个 TPB 并发请求，CPU 顺序仲裁）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CCB 为专用控制 IP（RISC-V X280 + 自定义 vector engine + DMA + 大 SRAM + 同步/中断），TPB cluster 为共享资源（指令缓冲/CPU/互连节点）的 4-TPB 组。使用：CCB 运行 NPU firmware（JIT 生成 TPB 指令）；cluster CPU 执行 CSU 触发的 scalar/vector/GSDU 服务例程，使 CPU 操作遵循与 tensor 操作相同的指令语义（简化编译/调度/派发）。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
