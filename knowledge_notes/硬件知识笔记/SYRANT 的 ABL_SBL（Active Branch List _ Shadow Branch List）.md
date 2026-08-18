## SYRANT 的 ABL/SBL（Active Branch List / Shadow Branch List）

术语解释
ABL/SBL 是 Prémillieu 与 Seznec（TACO 2012，SYRANT）提出的重收敛检测硬件：ABL 记录动态分支的活跃列表，误预测时其后条目转入 SBL（影子列表），resolved path 与 SBL 顺序比对检测重收敛后，用 SBL 中 squashed 分支的 outcome 在 fetch 阶段覆盖预测器——本文将其实现为对照 baseline。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) SYRANT 主体是对称资源分配（taken/not-taken 两条路径分配相同的物理寄存器/ROB/LSQ 资源，使重收敛点之后的 CIDI 指令资源可复用）；(2) ABL/SBL 是其附带的重收敛检测机制：分支执行后 outcome 收集进 ABL，误预测时 ABL 中误预测分支之后的条目复制到 SBL；(3) resolved path 上每个新分支插入 ABL 时与 SBL 比对，检测到重收敛后，SBL 中带 outcome 的条目用于在 fetch 阶段覆盖预测器（"SBL prediction"）；(4) 作者报告在很多应用中 ABL/SBL 预测质量超过 TAGE，可用单个 4-bit 全局计数器做开关。本文将其 fetch 阶段 squashed-branch reuse 单独实现作为 SOTA 对照：ABL 592 项（在飞指令上限）、SBL 512 项（同 ROB）、并"优惠"地复用了本文的每分支置信计数器。本文指出 SYRANT 的三条局限：(i) 两条防错位约束（重收敛搜索止于 resolved path 上第二个误预测分支实例；SBL 只搜到第二个实例）在 loop branch 误预测的 under-iteration/over-iteration 场景直接放弃复用机会；(ii) 首次 divergence 后复用必须终止（论文未交代处理）；(iii) SBL 只保留最近一次 squash 的 outcome（乱序窗口内多次 squash 会丢信息）。对比本文 SBRB：不变签名实现 near-perfect 对齐、无两条约束、容忍 CI 区任意 divergence、天然跨多次 squash。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（loop branch 误预测，predictor 提前退出循环 under-iteration）：squash 时 SBL 装有"循环后若干分支"→ resolved path 重取循环、下一个实例的 loop branch 出现时，SYRANT 约束 (i) 使重收敛搜索终止（第二个 mispredicted branch 实例），无法继续 latch-on，后续分支无法被 SBL 覆盖；而 SBRB 在进入下一次外层迭代（签名 A.2 前缀）后所有分支按 key 独立命中，与重收敛搜索无关。性能数据（同模拟器、同 benchmark、同置信机制）：SBRB geomean SPEC +2.08% / GAPBS +7.25% / 全部 +4.43%，SYRANT 分别为 +1.27% / +2.36% / +1.77%；SBRB 在除 401.bzip2（非循环控制流）外所有 benchmark 上胜过 SYRANT。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ABL/SBL 实现：ABL 是 fetch 单元按序插入的环形列表，每项记录分支信息与 outcome；SBL 为影子副本，仅在 squash 时写入、随重收敛对齐推进；置信用单一全局 4-bit 计数器（本文替换为每分支计数器以做公平对照）。SYRANT 完整系统还包含 RANT 表（gap-size 跟踪）与 gap 插入逻辑，用于对称资源分配。定位：ABL/SBL 是"顺序比对式"对齐的代表（同类的还有 Akkary branch recycling 的 PC 关联查找、Galliher Reuse Queue 的迭代计数器重同步），本文的不变签名是"标识符式"对齐——这也是本文 Related Work 中"对齐难"论点的核心对比对象。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
