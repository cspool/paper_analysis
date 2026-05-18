## Virtual Bank (VBA) — Row Granularity Memory Bank Organization

术语是什么？

Virtual Bank (VBA) 是 RoMe 提出的一种新的 DRAM bank 组织抽象。在传统 HBM 中，MC 必须分别管理 bank group、pseudo channel 和 individual bank 三层结构来做 bank group interleaving 和 PC interleaving 以最大化带宽。VBA 将这些细节从 MC-DRAM 接口中移除，由两个来自不同 bank group 的 physical bank 以 time-multiplexed 方式组成，使得单个 VBA 即可独立提供满通道带宽。最终采用 VBA = 2 banks from different BGs (Figure 7d) + 2 PCs operate concurrently (Figure 8b)，无需修改 DRAM 内部数据路径宽度，有效 row size 从 1KB 升至 4KB，banks/channel 从 128 降至 32。

从芯片设计角度拆解术语：

VBA 的核心芯片设计决策在于避免增加 internal dataline 宽度。论文探索了 6 种 VBA 配置（3 种 bank group 消除方式 × 2 种 PC 消除方式）。其中 Figure 7(b)（单 bank 通过加倍 AGbank 充当 VBA）需加倍 BK-BUS 和 internal bank dataline，且结合 Figure 8(a)（单 PC 加倍 fetch size）后总 dataline width 变为 4×，面积开销可达 77%。最终采用的 Figure 7(d)+Figure 8(b) 方案利用现有 DRAM 结构：两个不同 BG 的 bank 在 BG-BUS 上按 tCCDS 间隔交替传输数据，无需加宽任何内部总线。在 HBM4 中，每 VBA 对应一个逻辑 bank address，MC 仅需选择空闲 VBA 即可发 RD_row/WR_row，不再感知 bank group 或 PC 边界。

术语一般如何实现？如何使用？

VBA 实现在 DRAM die 内部，对 MC 透明。MC 按 4KB row granularity 发 RD_row/WR_row 后，HBM logic die 中的 command generator 负责将 row command 展开为对 VBA 内两个 physical bank 的 ACT/RD/PRE 序列：先对 Bank A 发 ACT（插入 tRRDS−tCCDS 有意延迟），再对 Bank B 发 ACT，之后按 tCCDS 间隔交替发 RD/WR。两个 PC 并发工作以双倍速率接收数据。每个 VBA 操作完成后自动回到 Idle 状态。VBA 设计的前提是 AGM_C 提升至 row size（4KB），因此不再需要 bank group 和 PC 来在不增加 AGbank 的情况下扩展带宽。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models
