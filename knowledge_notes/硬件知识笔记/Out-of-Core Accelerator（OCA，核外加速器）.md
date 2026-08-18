## Out-of-Core Accelerator（OCA，核外加速器）

术语解释
集成在 CPU 核流水线之外的加速器抽象：直接读内存输入、写内存输出，任务可留内部状态；典型代表是挂 LLC/片角的 Intel DSA、IAA、NPU 与 AMD APU 中的 GPU。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OCA 是"加速器作为独立设备"的形式化描述。优势：配专用 out-of-core 访存硬件（work queue、descriptor 引擎、大流缓冲），内存访问性能远高于核的 LSQ 路径。致命缺陷：任务可写内存、可留状态，错误推测的执行效果无法撤销，因此 OCA 不能被推测/乱序调用——调用（MMIO store 写工作描述符）必须等指令到 ROB 头才发出；ROB 头一次只能有一条指令，故两个 OCA 任务的"启动"无法重叠；调用 store 与后续轮询完成记录的 load 之间必须 fence 防止 store→load 重排，而 fence 会连带挡住无关的普通 load。论文量化：小任务（512B–2KB 解压任务）细粒度交错时，LLC OCA 比 ATX NCA 慢 9.4×（8KB 任务）到 2.6×（128KB 任务）；任务足够大时两者接近，但要以牺牲交错粒度和增大片上 scratchpad 为代价。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 Intel DSA 为例（论文 OCA baseline 原型）：软件填好 64B 工作描述符（src/dst 地址、长度、操作码、完成记录指针）→ 用 `MOVDIR64B`（posted）或 `ENQCMD(S)`（非 posted，反馈 retry）把描述符写进 MMIO portal（BAR2，每 portal 4KB 页）→ DSA 硬件把描述符搬入 work queue、DMA 完成数据搬移 → 更新完成记录 → 软件轮询完成记录状态位。论文 OCA baseline 的关键差异：其 out-of-core 访存接口换成了 UTE 的流引擎（继承 stream 支持），只保留 OCA"写内存 + 非推测调用"的语义缺陷，从而把对比聚焦在调用模型本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
落地形态：Intel DSA（数据搬移/CRC/DIF）、IAA（解压/压缩/扫描）、DLB（负载均衡）、NPU，AMD APU 集成 GPU；Linux 经 `idxd` 驱动暴露，用户态可用设备接口（devdax）或库（Intel DML/QPL）。编程模型本质是"共享内存 + 描述符队列 + 完成轮询"。适用场景：粗粒度、长任务（大块搬移、压缩）、写密集（scatter/memcpy）——这些 NCA 做不了（NCA 不写内存）；OCA 还能被多核共享（NCA 每核私有）。不适用：小任务、需要与核计算细粒度交错的场景。

涉及论文标题：
- ATX: Accelerator Task Extensions
