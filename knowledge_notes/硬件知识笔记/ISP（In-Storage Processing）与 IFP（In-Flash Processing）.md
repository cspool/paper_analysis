## ISP（In-Storage Processing）与 IFP（In-Flash Processing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ISP（in-storage processing）在 SSD 控制器上处理数据；IFP（in-flash processing）在 NAND flash die 内处理数据。GRAINS 的 ISP 单元（on-controller）：(i) 流式扫描 Colors Bitmap——从 NAND 流直接操作、每 channel 仅 2 个 32-bit 寄存器（当前+下一个 bitmap chunk），把结果立即消费，不缓存进内部 DRAM，避免 DRAM 带宽瓶颈（如 [420] 的流式做法）；(ii) 从内部 DRAM 流式取 unitig ID、调度 IFP 操作、与 host 和内部 DRAM 交互；FSM 协调执行流。GRAINS 的 IFP 单元（on-die）：实现 selection（按偏移从页缓冲抽取目标窗口，复用/重定向已有列解码与列选择电路，把目标窗口路由到 die 的 I/O 接口或内部比较单元）与 comparison（小移位寄存器+位比较器，把 k-mer 与目标窗口逐位匹配）两种轻量功能；flash 控制器经通道总线把参数（偏移或 k-mer）发给 on-die PE 的小寄存器，随标准页读命令下发，PE 在页读入页缓冲后操作，控制器只读回结果（而非整页）。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- IFP 高层流程：页读入 die 页缓冲 → 内容经 ECC_LITE 纠错 → PE 对页缓冲内容操作。Offsets 查询（图 10）：host 发 Offsets 索引+压缩 k-mer（chunk）→ 内部 DRAM 双 2-MiB chunk（一入一查）→ 按 GRAINS 映射转物理地址 → 因 host 排序+uniform 布放，访问为顺序流、全利用内部带宽 → 页入缓冲 ECC_LITE → IFP selection 只提取目标 Offsets 条目回传 → DRAM 缓存供 Strings。Colors 查询（图 12）：ISP 从 DRAM 流式取 unitig ID（1）同时并发扫描 Colors Bitmap（2）→ 遇 "1" 递增 Color Index（3）→ 位图位置匹配 unitig ID 时用 Color Index 索引 Colors（4）→ 同 Offsets 方式 IFP 取相关页片段（5/6）。效果：不把低复用/未用页传出发光 die，避免 channel 与外部 I/O 带宽浪费。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog 实现 + 22nm 库 Synopsys Design Compiler 综合（333 MHz，面积功耗极小：on-controller 0.0025 mm²/0.21 mW；on-die ECC_LITE 0.036 mm²/18.04 mW、其他 on-die 逻辑 0.000093 mm²/0.01 mW；ECC_LITE 占 on-die 逻辑面积 99%）；也可运行在通用 ISP/IFP 引擎上（近期采用）。评估：MQSim（SSD 内部操作）+ Ramulator 2.0（内部 DRAM）+ 自研组件模拟器组合；IFP 吞吐受 flash 读瓶颈限制，333 MHz 已足够。GRN（完整 ISP/IFP）比 GRN-Ext（同优化、逻辑在 SSD 外、PCIe 16 GB/s）平均再快 2×，证明 die 内/存储内处理消除外部带宽瓶颈的价值。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
