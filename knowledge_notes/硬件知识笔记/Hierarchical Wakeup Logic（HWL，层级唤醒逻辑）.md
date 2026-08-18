## Hierarchical Wakeup Logic（HWL，层级唤醒逻辑）

术语解释
本论文提出的 IQ 唤醒逻辑结构：把 IQ 逻辑划分为多个 segment，每个 segment 配一个小型、快速、非流水化的 level-1（L1）唤醒矩阵，其后放置一个全尺寸、流水化的 level-2（L2）唤醒矩阵；同 segment 依赖用 L1 单周期完成 wakeup-select，跨 segment 依赖用流水化 L2 唤醒，从而把 IQ 周期缩短 53% 而 IPC 仅退化 0.9%。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 动机——IQ 唤醒矩阵延迟随窗口尺寸平方增长，通用流水化 wakeup-select 又破坏依赖指令 back-to-back 发射，故需新结构在降周期的同时保住 IPC；(2) 结构——IQ 逻辑切 8 个 segment，每 segment 一个物理独立的小 L1 矩阵（默认 25 项），L2 为全尺寸 200×200 矩阵但 3 级流水化；(3) 唤醒规则与派发无关——producer 与 consumer 同 segment 用 L1（1 cycle），否则用 L2（3 级流水，多 2 周期）；(4) 周期 = max(L1+select, L2 单级)；(5) L1 容量有限 → 需 HSD 派发把依赖留在同 segment、混合模式处理容量竞争；(6) 与 H-SW 区别——H-SW [11] 把最老未就绪指令移到快速小 IQ（随机队列中"找最老"复杂、不考虑 L1 容量效率，IPC 退化 3.6%）；与 narrowing [3] 区别——narrowing 按依赖距离无条件派发到 producer 附近、无 L1 容量管理（退化 3.2%）。Web 证据：依赖矩阵唤醒（Goshima 等 MICRO-34）与层级调度窗口（Brekelbaum 等）均为本方案直接相关先例，本论文在相同周期时间下与其对比 IPC。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（Fig.3/4）：HSD 在 rename 时决定目标 segment（RMT 读 producer segment 号，LRP 预测 last-ready 源）→ 同 segment 则在 L1 写 cell（列=segment offset），跨 segment 则在 L2 写 cell → 执行时 producer grant 驱动 L1 或 L2 的 wordline，消费者 ready = L1 OR L2 → select。额外硬件开销：L1 矩阵共 0.6KB、LRP 2.0KB、segment 分配电路（DCL/LBMUX/SMUX1/SMUX2/SMCL，10-wide 下为 STA 的 1.59×，可再流水化 rename 代价 +1 cycle 误预测惩罚即 -0.6% IPC）、RMT 加 segment/offset/长延迟字段。默认 (N,P)=(25,3)：周期降 53%、IPC 退化 0.9%；1.5× 均衡 scaling（300-entry IQ）下 IQ 延迟仅为常规 200-entry 基线的 88%，支撑平均 +17.2%（最大 +43.1%）IPC。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/评估：在 SimpleScalar 3.0a 自建模拟器（Alpha ISA，SPEC2017 100M 指令 SimPoint 区域）中建模分段 IQ、L1/L2 矩阵、HSD、LRP 与混合派发，测 IPC；用 HSPICE（22nm ASU PTM、ITRS 线参数、MOSIS λ 晶体管级版图）测矩阵与 select 延迟得周期时间。论文（ISCA 2026 pp.529-542，Ando & Shimada，名古屋大学）未提供代码/artifact 链接，联网未找到开源仓库，无法确认开源；底层 SimpleScalar（github.com/toddmaustin/simplesim-3.0）与 ChampSim（github.com/ChampSim/ChampSim）开源可作复现基础。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
