## DRAM 行缓冲与行切换（Row Buffer / Row-Changing，tRCD/tRP/tCCDL）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DRAM 行缓冲（row buffer / sense amplifier 阵列）是每个 DRAM bank 内一次 activate 打开的行数据暂存区：读操作把整行从存储单元阵上电到行缓冲（activate，耗时 tRCD），此后行内连续列访问成本低（列命令 tCCDL 节奏）；访问另一行必须先 precharge（tRP）关当前行再 activate 新行，这一"行切换"远贵于行内连续访问。因此 DRAM 性能优化的首要目标是提高行缓冲命中率、减少行切换。FlexQ-NDP 的关键常数（DRAMSim3，GDDR6）：tCK=0.66ns、tRCD=24、tRP=24、tCCDL=4、BL=16、tCL=24——一次行切换的空闲窗口 t_RP+t_RCD=48 cycle，而一次 dequant 仅 8 cycle（2·t_CCDL），这是"去量化隐藏"技术成立的时间余量。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 FlexQ-NDP 的 GDDR6-AiM 风格芯片中：每个 bank 一个 PU + 5Kb SRAM，PU 从本 bank 行缓冲读数据（256b 列宽对齐 GDDR6 位宽）。布局直接决定行切换次数：baseline 把 scale 与 value 分离存储 → 计算中交替访问两个区域，每次切换触发 precharge+activate（且来回切换要再切回原行），行切换比粗粒度 INT 高 2.4~11.6×、延迟 +1.34~4×；scale-value 交织布局把 scale 区与 value 区相邻排布、对齐循环访问节奏 → 行切换降约 2×。代价模型把行切换细分为"自然行切换"（顺序流完一行到下一行）与"缓冲 miss 引发行切换"（读激活要切行、读完再切回权重行 = 2 次额外切换；与自然切换重合则只算 1 次；两个缓冲同时 refill 也省 1 次），用解析式 Lat = t_CCDL·Num_col + Lat_RowChange·Num_row + Lat_Dequant·(1−Ratio_Overlap) 估延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：行缓冲是 DRAM 芯片 bank 内固有物理结构（sense amplifier 阵列），芯片设计层通过 bank 数/行大小/列宽与命令时序（tRCD/tRP/tCCDL）定义其行为；软件层通过数据布局与访问顺序适配。使用：NDP 编译器的 DRAM mapping 步骤把逻辑地址映射为物理 (row, col)，使时间相邻的访问落在同行；仿真用 DRAMSim3 时序参数建模。对 NDP 的通用启示：行切换是 NDP 数据布局优化的核心成本项——布局的量化标准就是"该布局下的行切换次数"。

P3-LLM 补充视角（ISCA'26，tCCD_S/tCCD_L 与 PIM 计算节奏）：P3-LLM 把 tCCD_S（same bank group 的列命令间隔，各代 HBM 约为 tCCD_L 的一半）作为 PIM 计算单元的工作节奏——低精度 PCU 面积小、频率可达 HBM-PIM（tCCD_L 节奏、4 个内存总线周期）的 2×，因此每 tCCD_S 发起一次 PIM 命令、读同一 256-bit 权重切片并让两个输入先后复用（Throughput-Enhanced PCU 的时间维输入复用），等效吞吐翻倍。能耗侧：TEP 以 tCCD_S 运行使 PIM 功耗 +28%（主要来自 DRAM cell 访问与列译码切换，这些不随权重复用而变化），但减少重复 DRAM row activation，整体能量效率 1.56× 更好。频率设定：NPU 1 GHz、PCU 500 MHz（对应 HBM2 的 tCCD_S=2 DRAM 时钟）。论文指出该 tCCD_S/tCCD_L 关系同样适用于 GDDR 与 LPDDR（LPDDR-PIM 的 tCCD_S 也为 tCCD_L 一半），TEP 可泛化到其他 PIM 器件。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
