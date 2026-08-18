## Dependence Checker（依赖检查器，GPU frontend OoO 冒险检测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dependence Checker 是 frontend OoO 方案（GhOST/sCROOGe）中检测 IsB 指令间数据冒险的组合逻辑部件：通过比较 IsB 指令的源寄存器（rs）与目标寄存器（rd）字段与 InFL/IsB 中的在飞目标寄存器，识别三类冒险并生成目标 IsB 条目的依赖位向量。逻辑链：frontend OoO 发射独立指令的前提是准确知道每条指令依赖谁——RAW（Read-After-Write）：3 个 rs 与 InFL 和 IsB 中所有 rd 比较；WAW（Write-After-Write）：选中 rd 与其他 rd 比较；WAR（Write-After-Read）：选中 rd 与 IsB 内 rs 比较（无需查 InFL，因为 Issue 之后按序执行，在飞指令不会在 IsB 指令读旧值之前被新指令覆盖）。sCROOGe 的 Dependence Checker 是纯粹的组合逻辑，每 cycle 处理一条新指令。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（Fig.4）：IsB 分配后的下一 cycle，Dependence Checker 对新指令操作——① 把该指令的 rs1/rs2/rs3 与 InFL 所有已分配项的 rd 比较（在飞 RAW）；② 与 IsB 所有已分配项的 rd 比较（IsB 内 RAW）；③ 把该指令的 rd 与 IsB 其他项的 rd 比较（WAW）；④ 把 rd 与 IsB 其他项的 rs 比较（WAR）；⑤ 把依赖结果写为该 IsB 条目的依赖位向量，供"独立/最老"位向量生成电路与 Issue Arbiter 使用。以 `I1: r1=ld(...); I2: r3=r1+r2; I3: r5=r6*r7` 为例：I2 的 rs1=r1 命中 I1 的 rd → I2 标为依赖 I1；I3 的 rs/rd 均不与 I1/I2 冲突 → 标为独立，被 Issue Arbiter 越过 I2 优先发射。依赖位在发射时按 IsB ID 清空（IsB 冒险不再适用），InFL 相应位按新发射指令的冒险更新。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：寄存器号比较器阵列 + 编码逻辑，输入 IsB/InFL 的 rd/rs 字段，输出每 IsB 条目的依赖位向量；为减少硬件，把独立/最老位向量生成做成分离的顺序电路。使用：作为 frontend OoO 的核心控制逻辑，配合 Issue Arbiter 实现"独立指令优先发射"；sCROOGe 评估中 frontend 方案最坏配置下仍能在 Issue 阶段 600ps（vs baseline 295ps，Table III）的时序内完成（1GHz 下不成为关键路径，关键路径在 Execute 993ps）。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
