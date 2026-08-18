## Token Tracker（token 依赖跟踪器：TS / TID / OR 表）

术语解释
DySHARP token-centric kernel fusion 的硬件支撑：三张表在 token/tile 粒度跟踪 Dispatch⇒GEMM-1⇒GEMM-2⇒Combine 依赖链的 readiness，供 megakernel 做 readiness-gated 调度。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
三表分工：(1) TS Table（Tile Status Table，1024 项，on-chip，可溢出 DRAM）——每项对应一个 tsize-token tile 与一行 GEMM TB，字段 Valid/ExpID/Row + DAcc（对该地址区域的 dymultimem.st 计数，达 tsize×bsize 判 Dispatch⇒GEMM-1 就绪）+ TBCnt1（本行 GEMM-1 TB 完成数，判 GEMM-1⇒GEMM-2 就绪）+ TBCnt2（本行 GEMM-2 TB 完成数，完成即触发向源 GPU 通知）；(2) TID Table（DRAM，低访问频率）——记录每个 token tile 的 token ID 列表（nToken + TID*，TPtr 索引）；(3) OR Table（Output Readiness Table，1024 项，on-chip）——每 token 一个 nReady 计数，收齐 topk 个专家完成通知即 Combine 就绪。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
事件流：dymultimem.st 到达 → DAcc++ → 达 tsize×bsize → 对应 GEMM-1 行 ready；GEMM-1 行内 TB 全部完成 → TBCnt1 满 → GEMM-2 行 ready；GEMM-2 行完成 → TBCnt2 满 → 经 TID Table 收集该 tile 的 token ID → 向各源 GPU 发完成通知 → 源 GPU 对 OR 表条目 nReady++ → nReady==topk → 该 token 可发射 Combine（dymultimem.ld_reduce）。一致性保证：所有表状态更新延迟到"写数据对所有 SM 可见"之后（检测到 ack，即数据已到达 LLC/DRAM）。实现：16-bank 双端口 SRAM（1R1W）满足并发读写，TS/OR 各 1024 项。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 token-centric scheduler 配套使用：scheduler 用专用 load 指令 spin-poll 就绪位；GEMM-1/GEMM-2 仅当 tracker 标记行 ready 且目标 SM 组有空位时发射；Combine 通信 kernel 查 nReady==topk 才发射 ld_reduce。tile 尺寸选 128（= GEMM tile 尺寸）：更小破坏 GEMM 计算利用率并增加同步开销，更大粗化重叠。硬件实现面积计入 GPU 侧 0.198mm²（TSMC 12nm 综合）。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
