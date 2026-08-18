## URAM（UltraRAM，AMD/Xilinx UltraScale+ 大容量片上存储原语）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- URAM 是 AMD/Xilinx UltraScale+ 架构 FPGA 的片上存储原语，每 tile 288 Kb，是单块 36 Kb BRAM 的 8 倍容量，密度高、端口/延迟特性与 BRAM 不同（权威参考 AMD UG573 UltraScale Architecture Memory Resources User Guide）。Alveo U55C 含 960 URAM tiles（~541 Mb）+ 2016 BRAM tiles（132 Mb）；Alveo U200 同为 960 URAM。在 Graph.hls 中 URAM 是顶点属性缓存与归约缓冲（reduce buffer）的实现载体——图处理的 vertex property cache 与 keyed reduction 需要大容量、高带宽片上存储，U55C 上 960 URAM 是设计空间探索的关键资源约束。
- 在芯片/架构中的作用定位：介于 BRAM（细粒度、低延迟、端口灵活）与 off-chip DRAM/HBM（大容量但带宽/功耗代价）之间的中容量片上层级，适合"大块数组 + 流式/随机访问"的图数据。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Graph.hls 生成的图加速器中的运转流程：每条 pipeline 内 8 个 PE、每 PE 一个 URAM-backed reduce buffer。72-bit URAM 行按属性位宽容纳 |72/bitwidth| 个值：32-bit 时 2 值/行 → 65,536 目的节点需 65,536/2/8PE=4,096 行=8 URAM/PE → 8×8=64 URAM/管道 → 14 管道×64=896/960（93%）→ 直接约束 L1 max partition size=65,536；16-bit 时 4 值/行 → URAM 需求减半、片上缓存容量翻倍；8-bit 时 8 值/行 → 16 URAM/管道、buffer 深度 4×。这就是 L2 位宽参数通过 URAM 账本驱动 L1/L2 依赖传播（forward pass 定硬件可行性）的机制。
- Web 佐证：Vitis 可用 `MEMORY_PRIMITIVE URAM`（PLRAM 配置）让 kernel 内存用 URAM；DLR 在 U55C 上 SAR 聚焦算法用到 608/960 URAM、1818/2016 BRAM，表明二者是稀缺关键资源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：HLS 中数组综合时可被映射到 URAM（array partition/`MEMORY_PRIMITIVE`），或 RTL 中例化 UltraRAM primitive；Graph.hls 由 GH-Architect 自动按位宽/容量推导 URAM 分配（无需用户干预），并在 DSE 中把 URAM 用量作为约束（选中配置不超 960 上限）。
- 使用：作为大规模顶点属性缓存/归约缓冲/预取结构；论文表明 URAM 容量决定可处理的 max partition size，从而影响分区策略（L3）与位宽（L2）的联合选择。跨论文复用：URAM 资源账本方法可推广到任何"片上大缓冲容量受限"的 FPGA 图/稀疏加速器设计。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
