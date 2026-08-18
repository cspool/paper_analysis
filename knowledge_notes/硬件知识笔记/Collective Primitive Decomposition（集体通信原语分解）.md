## Collective Primitive Decomposition（集体通信原语分解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Collective Primitive Decomposition 是 RoCC 论文提出的、把集体通信（CC）操作语义与 ROP 硬件执行语义之间的"语义鸿沟"桥接起来的分解方法。核心洞见：所有 collective 例程都能拆成少量基本原语（primitive，借鉴 NCCL 的 send/recv/recvReduceSend/recvReduceCopySend/recvCopySend 等），每个 primitive 又能进一步拆成 ROP 可执行的 μOp（微操作）序列。RoCC 定义 5 种 primitive（send、recvReduceSend、recvReduceCopySend、recvCopySend、recv）与 5 种 ROP μOp（ReadDoorbell、Write、DepBarrier、Add、RingDoorbell），并用两个查找表译码器实现两级翻译：collective decoder 把 CC 操作按当前 stage 查表转成 primitive 序列（3 操作×15 阶段×3-bit=135 bit），primitive decoder 把每个 primitive 转成至多 6 个 μOp 的序列（5×6×3-bit=90 bit），共 225 bit（投影 1KB 查找表）。例如 4-GPU ring AllReduce 被分解为 send → recvReduceSend×2 → recvReduceCopySend → recvCopySend×3 → recv 的 7 阶段 primitive 序列，其中 recvReduceCopySend 展开为 ReadDoorbell → DepBarrier → ReadDoorbell → Add → Write → RingDoorbell 六个 μOp。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 ROP 中的运转流程（RoCC 论文 Figure 18）：① doorbell 到达后，collective decoder 依据 RoCC descriptor 中的 CollType 与 doorbell 的 Stage 字段查表生成当前 primitive → ② primitive 送 primitive decoder 译成 μOp 序列 → ③ μOp 进 collective command buffer（4 条目，每执行单元一条，66B）→ ④ 命令生成器每 cycle 发 4 个 μOp 到 4 个 ROP 执行单元 → ⑤ 每个 μOp 沿 ROP 既有数据通路执行：ReadDoorbell 用 ROP 内存 load 取上一 rank/本地数据、DepBarrier 检查 tile 完成（比较 doorbell 的 Offset/Stage）、Add 用 ROP ALU 归约、Write 用 ROP store 写回、RingDoorbell 经 doorbell manager 构造门铃包发往下一 rank。一个 primitive 完成后 doorbell manager 递增 Stage，按 CC 类型决定下一 rank。示例（AllReduce 第 3 阶段 recvReduceCopySend）：Rd→DepB→Rd→ALU→Wr→Rng，先读上一 rank 归约结果、等本地 GEMM tile 完成、再读本地结果、ALU 相加、写回、RingDoorbell 转发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：两级查找表译码器 + 4 条目 collective command buffer，是 RoCC 新增的 ROP 扩展逻辑（面积贡献：1KB 查找表 + 66B 命令缓冲）。使用方式：任何 CC 操作（AllReduce/AllGather/AllToAll）只要预先把"CC→primitive→μOp"映射写入查找表（本文支持 4/8-GPU ring 算法，8 GPU 时最多 15 阶段），ROP 即可自主逐阶段执行，无需 SM/CPU 调度。作用：使无 collective 语义的 ROP 仅凭 5 种原子/访存类 μOp 就能完成任意多步 CC，是"复用既有 ROP 硬件做 CC"的语义基础；本文还证明该分解表仅 225 bit（投影 1KB），硬件代价极小。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
