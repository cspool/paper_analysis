## Command Generator (in DRAM Interface)

术语是什么？

Command Generator 是 RoMe 引入的硬件模块，放置在 HBM logic die 中，负责将 MC 发出的简化 row-level 命令（RD_row/WR_row）转译为传统 DRAM 所需的完整 command sequence（ACT → 多个 RD/WR → PRE）。与传统 MC 中的 command scheduler 不同，command generator 不做动态调度——它按预定的固定时间间隔静态发出 DRAM 命令，不检查 bank state，不处理 timing constraint violation。每个 legacy channel 对应一个 command generator。

从硬件架构角度拆解术语：

Command generator 的固定展开逻辑（以 RD_row 为例）：接收 RD_row → 解析目标 VBA（对应两个 physical banks from different BGs）→ 插入 tRRDS−tCCDS 延迟 → 对 Bank A 发 ACT → 等待 tRCDRD → 对 Bank B 发 ACT → 等待 tRCDRD → 在 tCCDS 间隔下交替向 Bank A 和 Bank B 发连续 RD 命令（每个 RD 读 32B，共需 128 个 RD 覆盖 4KB row + 2 banks）→ 发 PRE 到 Bank A → 发 PRE 到 Bank B → 返回 Idle。WR_row 序列类似但使用 WR 命令和 tRCDWR/tWR timing。对于 refresh：MC 每 2×tREFIpb 发一次 per-bank refresh 命令，command generator 对 VBA 内两个 bank 间隔 tRREFD 各发一个 REFpb，将每 VBA stall 时间从 2×tRFCpb 降到 tRFCpb+tRREFD。

术语一般如何实现？如何使用？

RoMe 中 command generator 以 Verilog 实现，经 Synopsys Design Compiler 7nm 综合，单个面积约 118.6µm²，36 个 channel 总计 4268.8µm²，仅占 HBM4 logic die 的 0.003%。能耗贡献平均仅 0.06% of total DRAM energy。Command generator 的 placement 有三种选项：MC 侧（不减少 C/A pins）、logic die（减少 MC-DRAM 间 C/A pins，但不减少 TSV）、DRAM die（减少 TSV 但需冗余放置）。RoMe 选择 logic die 作为中间方案。若用于 non-HBM DRAM（如 DDR5），由于没有 logic die，需调整 placement 策略。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

