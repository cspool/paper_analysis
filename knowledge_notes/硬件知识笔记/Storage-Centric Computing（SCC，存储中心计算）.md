## Storage-Centric Computing（SCC，存储中心计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SCC 指在存储设备内部处理数据的范式，分两种实现位置：ISP（in-storage processing，在 SSD 控制器上计算）与 IFP（in-flash processing，在 flash die 上计算）。三大动机：(i) 消除低复用数据在存储系统与计算单元/主存之间的搬运；(ii) 把低复用数据处理的负担从主机 CPU/DRAM 移走，使其可用于其他任务或关闭省电、简化降成本；(iii) 利用 SSD 高内部带宽（现代 SSD 内部带宽常远超外部：例控制器 14 GB/s 外部 vs 57.6 GB/s 内部，经 16 channel 每通道峰值 3.6 GB/s；IFP 聚合全部 die 的带宽更高）。GRAINS 是第一个面向大规模基因组图分析的 SCC 系统，针对基因组图独特的随机、依赖、低局部性访问做存储感知算法-架构协同设计（batching/reordering + IFP + GST 调度 + FTL 改造），避免"直接把现有工具搬进 SSD"在随机访问下导致的 channel/die 争用与内部带宽浪费。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- GRAINS 的 SCC 执行流（图 6）：① host 发 GRNS_Start 命令，SSD flush 常规 FTL 元数据、加载 GRAINS 元数据进入 SCC 模式（图数据/查询以块粒度 round-robin 均匀布放 16 channel × 8 die × 4 plane，active block 跨 plane 页偏移对齐）；② host 完成查询准备（batching/reordering）后经 GRNS_Steps 触发 FSM，批次经标准 NVMe 数据路径进 SSD 内部 DRAM（不写 flash）；③ SSD 内：Offsets 顺序访问（IFP selection）→ GST 调度 round-robin 访问 Strings（IFP comparison，multi-plane）→ ISP 流式扫 Color Bitmap；④ 仅结果回 host。SCC 期间无写，故无 GC/磨损均衡，块粒度 L2P（1 TB 图 0.7 MB）释放内部 DRAM 给调度表；GRNS_Write 专门写基因组图（同时更新常规与 GRAINS 的 L2P）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现于现代 NAND 闪存 SSD：轻量专用 ISP/IFP 单元（Verilog + 22nm Synopsys DC 综合：on-controller 0.0025 mm²/0.21 mW，占 SSD 控制器 4 个 ARM 核面积 0.7%；on-die ECC_LITE 0.036 mm²/18.04 mW 占 flash 芯片面积 0.2%）或通用 ISP/IFP 引擎（如 [406-421] 的通用 ISP、[403,404] 的通用 IFP），非 SCC 模式 SSD 行为与普通 SSD 一致。评估：MQSim（SSD 内部）+ Ramulator 2.0（内部 DRAM）+ 自研组件模拟器 + 真实系统（EPYC 7742 + 1.5 TB DRAM）实测 host 段。结果：GRN 2.7×–47.8× 速度（4.4×–31.6× 能耗降）vs 软件基线 Fulgor/MetaGraph，1.5×–17.0× vs 理想硬件加速基线 IdealAccMem；低配 64 GB DRAM 系统 GRN($) 仍超高配 FG($$$)/MG($$$) 4.7×/5.2×。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
