## DIMM-based NDP（DDR5 DIMM 近数据处理：RCD / DB / rank / sub-channel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DIMM-based NDP 是把近存加速逻辑放置在双列直插内存模组（DIMM）上的 NDP 形态：在标准 DDR5 DIMM 中，一个 RCD（Register Clock Driver，寄存时钟驱动器）维护命令/地址（CA/CLK）信号完整性，多个 Data Buffer（DB，数据缓冲）维护数据（DQ/DQS）信号完整性，ranks 内含高密度 DRAM 芯片存数据。DDR5 的关键组织特征：每个 rank 由 2 个独立 sub-channel 组成（各 32 bit 数据 + 8 bit ECC），每 sub-channel 若干 DRAM device 同步工作；sub-channel 之间无直接通信路径、跨通道数据交换必须经处理器中转（代价高）。NASZIP 把近存加速器 NMA（VPE + LNC + 共享优先队列 + 控制器）集成进 DB 芯片、不改标准 DRAM die，从而保留 host 兼容并复用处理器 DDR 控制器（Web 证据：DDR5 RCD/DB 架构与 MRDIMM/MRCD 商品化——Renesas RG5R256/RG5R188、Montage M88MR5RCD01 等；vault 笔记：/data3/paper_analysis/knowledge_notes/芯片知识笔记/DRAM-Based Processing-in-Memory (PIM).md 给出 DRAM PIM 的 bank 级计算组织与约束）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
芯片组织（论文 Fig.2、Fig.10）：host CPU 经内存通道连到多个 DDR5 DIMM；每 DIMM 含若干 rank，每 rank 2 个 sub-channel、每 sub-channel 4 个 8-bit DRAM device；NASZIP 每 sub-channel 集成一个 NMA（VPE+LNC），控制器、共享优先队列、两个 LNC 与两个 VPE 封装在 buffer chip（DB）旁。运转流程：CPU 经 DDR 控制器发距离计算命令 → RCD 路由 CA → NMA 按 DaM 从本 sub-channel 取 Dfloat 向量（4 device 并行、每 burst 128 bit）→ VPE 算距/早退 → 两 VPE 结果经共享优先队列合并排序 → 只回传 top 候选给 CPU。芯片级设计要点：① 每向量完整落在一个 sub-channel、维度跨 4 device 交叉，burst 数与 Dfloat 段位宽对齐（SIFT 128 维 18/14/16 bit → 6/4/6 burst）；② 面积 0.7091 mm²（28nm 综合）vs 标准 RCD/DB 10.22 mm²，开销可忽略；③ 热：3D-ICE 评估峰值 DRAM 温度 65.47°C（环境 28°C），低于 JEDEC 默认刷新 t_REFI 的 85°C 阈值，无需主动散热；④ ECC：on-die ECC 在 DRAM die 内部、不受 NMA 影响，side-band ECC 因 Dfloat 只是软件表示、物理仍是标准 DDR5 burst 格式而兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现路径：RTL 实现 NMA 逻辑（FPGA 验证功能）→ Synopsys DC 28nm 综合 + Cadence Innovus P&R 得面积/功耗 → UniNDP 周期精确模拟器评估 QPS/延迟/recall（模拟器按 DDR5-4800、19.2 GB/s per sub-channel 配置）。业界方向：MRDIMM/MCRDIMM（Multiplexed Rank RCD/MDB，Renesas RG5R188、Montage M88MR5RCD01，8800 MT/s）把命令去交织与 rank 级路由做成商品芯片，是 DIMM 上集成更多逻辑（含近存计算）的演进路径（Web 证据）。论文配置：DDR5-4800、2 或 6 通道、2 DIMM/通道、2 rank/DIMM、2 VPE+LNC/rank、1.2 GHz；6 通道 48 sub-channel 聚合带宽 921.6 GB/s。开源：NasZip 仓库含 RTL 与 UniNDP 模拟配置。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
