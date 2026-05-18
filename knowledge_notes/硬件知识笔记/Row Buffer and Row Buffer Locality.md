## Row Buffer and Row Buffer Locality

术语是什么：

DRAM Row Buffer（也称为 sense amplifier row buffer）是每个 DRAM bank 内部的锁存器阵列，宽度等于一行的大小。当 ACT 命令激活某行时，该行所有 cell 的电荷被 sense amplifier 放大并锁存到 row buffer 中。随后的 RD/WR 命令只需从 row buffer 中选通对应 column 的数据，速度远快于直接从 cell 读取。Row buffer locality（行缓冲局部性）指连续访问命中同一 row buffer 的比例——高局部性意味着大量访问共享一次 ACT 开销，对 DRAM 带宽利用率至关重要。

从硬件架构角度拆解术语：

HBM4 中典型 row size 为 1KB（256b AGbank × 32 banks 共享的 data path 结构）。在传统 column-level 访问下：ACT → 1KB row 全部进入 row buffer → 随后每个 32B RD 从 row buffer 中选通对应 column → 多个 RD 可连续命中同一 row buffer → 访问结束后 PRE 关闭行。Row buffer hit 意味着后续 RD/WR 仅需 tCCDS（~1ns）而无需 tRCD（~16ns）的 ACT 延迟。RoMe 通过将访问粒度提升到 4KB（整行），使每次 RD_row 本身就覆盖整个 row buffer——因此 row-buffer locality 被 row granularity 自动保证，MC 不再需要跟踪 open row 和做 hit/miss 判断。但同时，由于有效 row size 升至 4KB（VBA 的两个 bank 各提供 2KB），可能对非连续访问产生 overfetch。

术语一般如何实现？如何使用？

Row buffer 是 DRAM 芯片内部的固有结构，无需额外实现。MC 通过 page policy（open/close/adaptive）来利用 row buffer locality：检测 pending requests 是否命中已 open 的 row，若命中则优先服务。GPU 和 AI accelerator 的 MC 通常采用 open-page policy 以利用 LLM 的高 row buffer locality。在 Ramulator 2.0 等 DRAM simulator 中，row buffer hit/miss 是 cycle-accurate timing 的关键建模要素。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

