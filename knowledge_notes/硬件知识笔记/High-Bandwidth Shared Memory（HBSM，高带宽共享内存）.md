## High-Bandwidth Shared Memory（HBSM，高带宽共享内存）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HBSM 是 TPB 内 2MB 高带宽共享 SRAM，同时充当数据存储与功能单元间灵活通信枢纽：生产者/消费者经预定义地址区间交换数据、以计数器同步，无需专用 datapath。采用 banked 设计：32 个 bank × 32B/cycle，地址按 32B 粒度交织，支持多单元并发访问；8 个 requester port；同 bank 冲突用 round-robin 仲裁并保证每 requester 保序。同步动作（标记数据 produce/consume）与内存访问绑定，仲裁获胜后该访问即全局可见。访问延迟约 20 cycle，但被流式执行掩盖（高带宽维持满吞吐）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：TCU/CVU/DTDU 经 TWU 生成的地址流并行读写 HBSM → 同一 bank 冲突由 round-robin 仲裁、每 requester 保序 → 写访问仲裁获胜时更新 SU 计数（数据生产全局可见）→ 消费者 monitor 计数达期望值后开始读 → 多单元吞吐由 32 bank 并行支撑。例子：卷积执行时 TCU 流式读激活/写部分和、DTDU 同时搬运下一 tile、CVU 处理中间结果，全部并发访问 HBSM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：多 bank SRAM 宏 + 仲裁器 + 与 SU 联动的同步逻辑；bank/port 数经建模与后端评估选择（32 bank、8 port 为面积-带宽最优平衡）。使用：编译器分配地址区间与同步计数，替代 register file 与显式 load/store 语义；数据流同步与数据搬移统一（"central backbone of the dataflow architecture"）。未开源。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
