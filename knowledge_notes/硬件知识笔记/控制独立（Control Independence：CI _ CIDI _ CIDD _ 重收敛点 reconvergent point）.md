## 控制独立（Control Independence：CI / CIDI / CIDD / 重收敛点 reconvergent point）

术语解释
控制独立指令是指位于误预测分支"动态重收敛点"之后、其执行不受该分支走向影响的指令；其中与分支数据无关者可直接跨 squash 复用，与分支数据相关者需重执行。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 分支 br1 有两条可能路径（取 B 或 C），两条路径在 label D 汇合，D 即 br1 的重收敛点（reconvergent point）；(2) D 之后的指令无论 br1 走哪条路径都会被取指执行，因此"控制独立"（CI）于 br1；(3) CI 指令再分两类：CIDD（Control-Independent Data-Dependent）——其数据来源受 br1 控制相关区域（{B,C}）影响，例如 br2 依赖 x，误预测后需选择性重执行；CIDI（Control-Independent Data-Independent）——数据也不受 br1 影响，如 br3，误预测后结果可直接复用；(4) 相对地，位于 br1 控制相关区域的指令为 CD（Control-Dependent）。本文的洞察：squash 时部分更年轻分支已乱序执行完（包括 br3 这种 CIDI 分支），它们的 outcome 在 resolved path 重取指时依然正确，可用来覆盖分支预测器——尤其对与 br1/br2 几乎没有相关性的 CIDI 分支，预测器很可能再次犯错。相关文献两大类：保留 CI 指令不被 squash 的微架构（TCI、Ginger、Skipper、Selective Branch Recovery 等，需复杂硬件替换 CD 指令并选择性重执行 CIDD 指令，且 out-of-order fetch 反而降低预测精度）；squash reuse 类（全部 squash 但保存 CI 结果供 rename 阶段复用：RI、MSSR、SYRANT 等，只降低 penalty 而不减少误预测数）。本文 SBRB 属于第三类方向：fetch 阶段的 squashed-branch reuse，直接消除误预测。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
控制独立为"哪些被 squash 的结果可以安全复用"提供判据，本文将其用于分支粒度：只复用 CIDI 分支的 outcome（对 CIDD 分支，squashed outcome 可能基于错误数据）。硬件不显式区分 CIDD/CIDI，而是用 BTB 中的每分支置信计数器间接学习"该分支的 squashed outcome 相对默认预测的可信度"：若某分支的 squashed outcome 常常与最终实际 outcome 一致（说明它倾向 CIDI 或数据变化不影响结果），计数器趋近饱和、允许覆盖；否则（CIDD 且数据影响大）计数器保持低位、不覆盖。运转流程例子：br1 误预测 → 阴影中 br3（CIDI）执行完写入 SBRB、br2（CIDD）执行但数据可能错误 → 重取指时 br3 命中 SBRB 且置信高 → 覆盖预测成功；br2 也命中 SBRB 但置信低 → 仍用默认预测器。SPEC 程序静态分支多、CIDD/CIDI 关系复杂且基线预测精度高，因此必须用每分支置信保护（全局计数器在 SPEC 上不可靠，在 GAPBS 上尚可）——这是本文置信机制设计空间的关键结论。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
控制独立在传统方案中的实现是"保结果"：RI（Register Integration）在 rename 阶段用 PC+源物理寄存器名查 Integration Table 复用被 squash 的物理寄存器；MSSR 用 WPB/squash-log 检测重收敛并复用；TCI 用 checkpoint 基底 + CIDD 重执行缓冲。本文的实现是"保分支 outcome"：SBRB + 不变签名对齐 + 置信门控，全部改动局限在 fetch 单元。CIDD/CIDI 概念还可用于判定哪些分支的 squashed outcome 需要在 SBRB 中保留（本文不显式淘汰，靠 key 与容量自然替换）。局限：非循环控制流（goto、手工汇编产生的 non-loop cycles）无法获得不变签名，本文观察到少数此类 benchmark（如 401.bzip2 的 mainQSort3()）复用效果差。

涉及论文标题：
- Augmenting the Branch Predictor with a Squashed-Branch Reuse Buffer
