## Doorbell 消息机制（门铃消息，SM↔ROP 与 ROP↔ROP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Doorbell（门铃）是 RoCC 论文提出的、连接 SM 与 ROP、以及不同 GPU 上 ROP 之间的轻量消息机制，让 ROP 无需 SM/CPU 参与即可自主完成多步 CC。消息载体是 doorbell descriptor（图 16b）：Offset（请求 warp 在目标 tile 中的地址偏移）、PayloadAddr（tile 指针地址/接收数据区地址）、SrcRank（来源 GPU rank）、Stage（当前 primitive 阶段）。ROP 内实现为两个模块：doorbell manager（识别 doorbell 消息并管理，7 状态状态机，仅需 2 个 1-bit 比较器 + 2 个 32-bit 寄存器 + 4-bit 计数器 + 4-bit 比较器）与 doorbell buffer（32 条目队列，每条目 3-bit 状态共 96 bit/ROP）。SM 发 RoCC 指令时在请求包头部置 doorbell flag，MPU 的 doorbell manager 识别后把请求信息复制进 doorbell buffer；跨 GPU 的 doorbell 包携带 tile 数据作 payload，接收方 doorbell manager 在 GPU 内存中预留的 doorbell region（每 MPU 4KB = 32 条目×128B，按 128B 内存交织粒度预留）里分配空间并复制 payload。每 doorbell 承载至多一个 tile（如 16×16×8 FP16 的 4KB 块，物理交织后每 MPU 分到 128B）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（RoCC 论文 Figure 15/18）：① 某 warp 完成 GEMM tile，执行 rocc_allreduce intrinsic → ② 带 doorbell flag 的指令包经原子数据通路到达目标 MPU → ③ doorbell manager 识别（与普通 atomic/L2 请求分流到独立 doorbell buffer，互不阻塞）→ ④ ROP 每 cycle 取至多 4 个 doorbell 分发到 4 个执行单元 → ⑤ collective/primitive decoder 按 CollType+Stage 译码执行 μOp → ⑥ 一个 primitive 完成后 doorbell manager 递增 Stage：非终态则构造新 doorbell 包（descriptor 编码进包头 + 计算后的 tile 作 payload）经 GPU-GPU 互连发往下一 rank 的 ROP；终态则写本地对称地址完成。示例：4-GPU ring AllReduce 中 ROP0 完成本地归约后 RingDoorbell 把结果发 ROP1，ROP1 的 doorbell manager 识别、stage+1、继续执行 recvReduceSend，共 7 阶段接力至全部完成。作用：把 CC 的控制流（下一步去哪、何时完成）编码进门铃，让 CC 完全脱离 SM/CPU 调度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：doorbell manager（状态机）+ 32 条目 doorbell buffer + 每 MPU 4KB doorbell region（GPU 内存预留，driver 在 tensor 处理前分配）。doorbell 识别通过比较目标地址是否落在本地 doorbell region 或检查包头 flag 完成；跨 GPU 发送复用 MPU 既有 memory issue 逻辑（4-bit 计数递增 Stage、4-bit 比较判终态）。使用方式：SM 端每条 CC 一条指令（ROP_AR/ROP_AG/ROP_A2A）即完成触发；ROP 端 doorbell 自动接力。硬件成本：doorbell buffer 0.75KB + doorbell manager 逻辑 + 每 ROP 96 bit 状态，合计约 L2 slice 面积的 2.4%（含其它 RoCC 组件，CACTI v7.0）。对比：NVMe 的 doorbell（写 SQ doorbell 寄存器通知设备）与 RoCC 门铃同属"轻量通知"思想，但 RoCC 把门铃扩展到"携带数据 payload + stage 状态"的设备间消息路由。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
