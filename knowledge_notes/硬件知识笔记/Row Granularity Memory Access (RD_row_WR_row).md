## Row Granularity Memory Access (RD_row/WR_row)

术语是什么？

Row Granularity Memory Access 是 RoMe 提出的 DRAM 访问接口范式，将最小访问粒度从传统 cache-line 大小（如 HBM4 的 32B）提升到一整行大小（4KB）。接口仅保留两个 row-level 命令——RD_row 和 WR_row——替代传统的 column-level RD/WR + ACT + PRE 命令组合。MC 不再发 column address，也不再管理 ACT 和 PRE 的时机。在 LLM 顺序大块访问场景下，一个 RD_row 替代 128 个 32B cache-line RD，大幅减少命令数量和 MC 调度开销。

从硬件架构角度拆解术语：

在 RoMe 系统中，RD_row/WR_row 的硬件执行流程：MC 收到 4KB memory request → address mapping 映射到 channel/SID/VBA → scheduler 选空闲 VBA → MC 发出 RD_row 命令 → HBM logic die 的 command generator 接收 row command → generator 按固定时序展开为：对 VBA 内 Bank A 发 ACT（经 tRRDS−tCCDS 有意延迟后）、对 Bank B 发 ACT、按 tCCDS 间隔在两个 bank 间交替发连续 RD 命令、发 PRE 关闭两行 → VBA 回到 Idle。数据通过两个 PC 并发传输（每 PC 2KB），合计 4KB。MC 只需管理 RD_row-to-RD_row/RD_row-to-WR_row/WR_row-to-RD_row/WR_row-to-WR_row 等 10 个简化 timing parameters（vs 传统 MC 的 15 个）。Row granularity 天然保证 row-buffer locality（每次访问整行），因此 RoMe 每次 access 后自动 precharge，不再需要 page policy。

术语一般如何实现？如何使用？

通过修改 HBM interface 标准和 MC 架构实现。需要：(1) HBM 协议层面定义 RD_row 和 WR_row 新命令；(2) DRAM die 侧（或 logic die）实现 command generator 将 row command 转译为传统 DRAM command sequence；(3) MC 侧简化为只发三种命令（RD_row/WR_row/REF），移除 column address 相关逻辑；(4) processor/accelerator 侧需能以 KB 级请求粒度访问内存（如通过 DMA engine 发 4KB request）。RoMe 通过修改 Ramulator 2.0 模拟器 + Verilog 综合（7nm）验证了该设计。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

