## HBM Logic Die

术语是什么？

HBM Logic Die 是 HBM 堆叠中的底部 die，位于 DRAM dies 之下，通过 TSV (Through Silicon Via) 与上层 DRAM dies 垂直互联。Logic die 负责缓冲和管理 DRAM dies 与外部处理器（GPU/TPU/accelerator）之间的数据传输，包含 GBUS controller、I/O ctrl buffer、PHY 接口等电路。与 DRAM dies 使用 DRAM process 不同，HBM4 的 logic die 使用 logic process 制造，因此可以集成较复杂的数字逻辑（如 RoMe 的 command generator）。

从芯片设计角度拆解术语：

HBM 堆叠架构中，多个 DRAM dies（HBM4 支持 16-Hi 配置）通过 TSV 连接到底部 logic die。每个 channel 的数据从 DRAM die 的 bank → BK-BUS → I/O ctrl buffer → BG-BUS → GBUS controller → logic die TSV → PHY → interposer → 处理器。Logic die 上每个 channel 有独立的 GBUS controller 和 PHY。由于 logic die 使用 logic process（而非 DRAM process），可以在其上放置额外数字逻辑（如 RoMe 的 command generator）而面积开销极小——RoMe 的 36 个 command generators 只占 logic die 的 0.003%。Logic die 也是 HBM 区别于 conventional DRAM（如 DDR5）的关键特征，后者没有 logic die 因此无法类似地集成 command generator。

术语一般如何实现？如何使用？

Logic die 由 HBM vendor 设计和制造，是 HBM stack 的一部分。在标准 HBM 中，logic die 主要做 PHY、buffer 和 test logic。RoMe 创新性地利用 logic die 的 logic process 优势，将 command generator 放置于此，从而在减少 MC-DRAM 间 C/A pins 的同时不修改 DRAM die 内部结构。未来若采用 hybrid bonding 等先进 die-stacking 技术，logic die 与 DRAM die 间的 TSV 成本可进一步降低，logic die placement 策略将更有吸引力。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models
