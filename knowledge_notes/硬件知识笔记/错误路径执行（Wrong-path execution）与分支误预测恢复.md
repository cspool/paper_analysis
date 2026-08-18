## 错误路径执行（Wrong-path execution）与分支误预测恢复

术语解释
错误路径执行是乱序处理器在分支误预测被确认前，沿推测方向继续取指/执行指令的现象（这些指令最终被 squash）；恢复（recovery）是误预测确认后回滚推测状态（ROB/重命名表/预测器状态）的过程。对 RBias 类基于值的预测器，错误路径上生成的值是否可用于后续预测、以及恢复机制的成本，是设计与评估的关键。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：(1) 乱序处理器为掩盖分支延迟而推测执行，误预测确认时 flush 掉错误路径指令并回滚；分支预测器本身的状态（如全局历史、局部历史、TAGE 表）也需要恢复（如 BADGR 用 GHR 恢复机制，Schlais & Lipasti ICCD 2016）；(2) trace-driven 模拟器（如 CBP simulator）不模拟错误路径执行——它只按 trace 顺序给预测器喂事件；execution-driven 模拟器（gem5）精确建模 flush、可观察错误路径对预测器可见状态的影响；(3) 对 RBias：digest table 保存"值 digest"或"在飞 ROB index"，flush 时 decode 阶段年轻指令写入的 ROB index 必须撤销、已完成执行的老指令 digest 必须保留——需要 checkpoint 恢复（Log-RBias，每 checkpoint ~1 Kbit、持续监听 completion 通知）；Seq-RBias 用 ring buffer 指针回滚替代 checkpoint（恢复只需恢复指针），且错误路径写入 buffer 的内容不清除——可在恢复后继续作为特征被后续分支预测利用；(4) 评估发现（Table III/Figure 14）：Log-RBias/noRecov（无恢复）因 digest table 被错误路径破坏而收益很小；Log-RBias 在 CBP simulator 与 gem5 上改善几乎一致（对错误路径效应近乎不敏感）；Seq-RBias 在 gem5 下收益显著大于 CBP simulator——它把错误路径执行从"状态破坏源"转化为"额外价值来源"。
- 关键程序场景（Figure 6/7）：fp_8_trace 中 R32>R33 分支的误预测不可避免，但产生 X1 的 load 不在 squash 范围内、X1 值及时可见 → 后续 X1>0/X2<0 分支被 RBias 正确预测，阻止误预测级联。Seq-RBias 还能利用 flush 后指针回滚但内容未清的"错误路径生成值"（Figure 7 quicksort 例子）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件位置：前端/后端交互的误预测恢复通路 + 预测器自身状态恢复结构。运转流程例子（Log-RBias 的 checkpoint 恢复）：分支 brX 误预测确认 → 流水线 flush、ROB 指针回滚 → digest table 恢复：执行单元广播 {ROB index, digest} 给当前表与所有 checkpoint，各 checkpoint 比较自己条目中的 ROB index、匹配则替换为 digest（保留已完成执行的更新）；decode 阶段年轻指令写入的 ROB index 被撤销（decode 更新只进当前表不进 checkpoint，flush 后当前表从 checkpoint 恢复）→ 恢复后的 digest table 只含"flush 点之前已完成执行"的值。
- 例子（Seq-RBias 的错误路径利用）：分支误预测 → fetch 回滚、ring buffer 指针回滚到 flush 点 → 错误路径上写入的条目内容仍在 buffer 中 → 后续分支预测时按动态距离观察到这些"错误路径生成的值" → 若与实际路径相关性高则提升预测（gem5 下 Seq-RBias 的 MPKI reduction 高于 CBP simulator 即源于此）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Log-RBias 的 checkpoint 结构（每 ~1 Kbit，数量取决于 flush 恢复粒度；每个活跃 checkpoint 持续监听所有 completion 通知——开销不 negligible 但可接受）与 Seq-RBias 的 ring buffer（fetch 顺序分配、指针回滚恢复、内容不清除；初步实验显示 buffer 可为 ROB 尺寸的 1/4 而无精度损失）。评估方法：CBP simulator（trace-driven，不建模错误路径）与 gem5（execution-driven，建模错误路径）双模拟器对照——Log-RBias 两者结果一致、Seq-RBias 只在 gem5 下展现错误路径收益。开源自 RUNLTS artifact（Zenodo 10.5281/zenodo.19453058）。

涉及论文标题：
- RUNLTS Branch Prediction with Register-Value Correlations and Hierarchical Table Orchestration
