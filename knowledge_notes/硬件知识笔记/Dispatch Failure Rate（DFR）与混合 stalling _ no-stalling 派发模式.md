## Dispatch Failure Rate（DFR）与混合 stalling / no-stalling 派发模式

术语解释
DFR（dispatch failure rate）是 HWL 中"因目标 segment 容量冲突而未能派发到 producer segment"的指令占派发总数之比；HWL 用周期性测量的 DFR 驱动运行时在 no-stalling 与 stalling 两种派发行为间切换（混合模式），以同时控制 L2 唤醒惩罚与 IQ 容量浪费。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 目标 segment 冲突时两个选项——no-stalling：派发到随机非目标 segment（IQ 容量利用高效，但跨 segment 走 L2 多 2 周期，惩罚累积）；stalling：阻塞等待目标 segment 出现空位（无 L2 惩罚，但阻塞期间 IQ 容量效率下降、派发吞吐受损）；(2) 单一策略无法覆盖所有程序：HSD-nostall 平均退化 1.7%、HSD-stall 平均 4.6%，均不满足 <1.0% 目标；(3) 混合模式——no-stall 模式下每 10k 周期测 DFR，高于阈值（默认 10.0%）切到 stalling；stalling 模式下每 200 个 max stall periods 回探一次 no-stall 以检查是否可恢复；(4) 难点程序分析（Fig.16 派发尝试分解）：xz chunk 大 → producer-segment 失败多（nostall 差、stall 好）；deepsjeng/xalancbmk/fotonik3d segment 竞争大或 IQ 常满（stall 差、nostall 好）；hybrid 按阶段自适应后平均退化 0.9%（Fig.15），但 xz/fotonik3d 仍有 5-6% 残留退化。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：派发时若目标 segment 无空位——no-stall 模式：随机选非目标 segment（随机选择在派发级前至少 1 周期投机完成，不增加派发电路关键路径复杂度，结果仅在需要时使用）；stall 模式：阻塞。计数器按 10k 周期窗口统计 DFR（失败数/派发总数）与模式状态机（Fig. 论文 IV-C）决定切换；DFR 越高说明 L1 命中率越低，此时 stall 等空位反而减少 L2 惩罚。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：派发级模式寄存器 + DFR 计数器（10k 周期窗口）+ 阈值比较器 + 状态机（no-stall ⇄ stall，200 间隔探测）；参数（DFR 测量间隔、阈值、max stall periods）为可调设计点（Table II）。使用：作为 HSD 之上的一层自适应控制，专门应对 chunk 偏大/竞争激烈的程序阶段；论文在 SimpleScalar 自建模拟器中实现三个模式并对比（Fig.15）。未开源。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
