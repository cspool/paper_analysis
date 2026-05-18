## Bank Group Interleaving

术语是什么？

Bank Group Interleaving 是 HBM（自 HBM2 起）引入的带宽扩展机制。DRAM bank group 将多个 bank 组合在一起共享 I/O control buffer 和 BG-BUS。不同 bank group 的数据传输可以在时间上交叠：当 bank group A 的数据正在 BG-BUS 上传输时，bank group B 可以同时准备下一批数据。MC 在不同 bank group 的 bank 之间按 tCCDS（通常为 tCCDL/2，如 1ns）间隔交错发出 RD/WR 命令，从而在 AGbank（per-bank 单次访问数据量）不变的条件下，使有效数据率达到两倍于 DRAM core frequency 的水平。

从硬件架构角度拆解术语：

以 HBM4 为例：BK-BUS 运行在 1/tCCDL 频率（~0.5GHz），每次传输 256b；BG-BUS 运行在 1/tCCDS 频率（~1GHz）。单个 bank group 只能利用一半的通道带宽，因为 BK-BUS 慢于 BG-BUS。要饱和通道带宽，MC 必须在两个 bank group 之间交替发命令：在 BG0 的数据通过 BG-BUS 传输的同时，BG1 的 bank 正通过 BK-BUS 准备下一批数据。MC 的 command scheduler 需持续跟踪所有 bank group 中所有 bank 的状态，寻找 ready bank 做高效交错。这要求 MC 维护大量 bank FSM 和请求队列。RoMe 通过提升 AGM_C 至 row size（4KB），使得单个 VBA 即可提供满带宽，从而从 MC 接口中消除了 bank group interleaving 的需求。

术语一般如何实现？如何使用？

Bank group 在 JEDEC HBM 和 DDR4/DDR5 标准中定义，由 DRAM 芯片内部硬件实现。MC 通过以下方式利用 bank group interleaving：(1) address mapping 将连续物理地址映射到不同 bank group 的 bank 上；(2) command scheduler 使用 bank group-aware 调度策略，优先选择与上一个命令不同 BG 的 ready bank；(3) 结合 page policy 决定何时 precharge。典型 MC 实现中，command scheduler 的 bank group interleaving + PC interleaving + bank interleaving 三层逻辑是面积和功耗的主要贡献者。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

