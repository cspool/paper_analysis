## FSM fast-forwarding（有限状态机快进）与计数器抽象（Counter Abstraction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FSM fast-forwarding 是 QED 提出的经验性状态空间缩减技术，用于对抗 RTL 形式化验证的状态爆炸；其思想推广自计数器抽象（counter abstraction，Jasper Design Automation 专利 [70]）：含计数器的设计只有少数关键计数值（如 0x10、0x20）触发关键逻辑，JasperGold 提供抽象让引擎在相关值之间"快进"而不穷举中间值（图 13 的 8-bit counter 例子）。QED 把它推广到 LSQ 的 load/store 条目 FSM（图 12：invalid→allocated→addr_ready→executed→succeeded→committed/observed 约 7 态）：验证某个直接序对时，其余 load 条目一分配就立即快进到终态 observed（红色 fast-forward 迁移，立即标记 executed/succeeded/observed），store 同理——因为这些其他指令对当前验证的序对不相关（每个指令对会被单独验证）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
不缩减时 N 个 load 的可达状态约 7^N，RTL 验证不可行。快进后，除直接序对外其余条目只有 invalid/observed 两态，且关键正确性论证：(1) 前缀-后缀不变式——valid 条目因按预测程序序连续分配形成 load queue 连续前缀、invalid 条目因 squash 作废违规指令及其后指令形成连续后缀（store queue 同理），故可达状态可数（论文加 SVA assert 验证该不变式）；(2) 缩减自身被验证——加断言确认 FSM 只含合法迁移、且一个条目的 FSM 不直接影响另一条目的 FSM（间接交互经外部模块（如队列填满）由 JasperGold 对全部未剪枝输入空间的穷举捕获）。注意与计数器抽象的区别：快进要操纵条目内多个字段，不能靠简单 JasperGold 命令，须加 shadow 逻辑修改 RTL 完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 BOOMv3 LSQ RTL 中加 shadow 逻辑实现 fast-forward 迁移（分配时直接写 observed 状态字段），配合 backward-slicing（剪 83% 输入、91% bit）与 assume 约束。效果：QED 在 128ld/64st 配置上无界验证 RVWMO 全部 227 个谓词，修正实现后 10 天内完成；扩展性曲线——LSQ 尺寸翻倍 full proof time 约×10（各谓词斜率一致）、proof depth 近似线性、memory 近似二次。使用要点：fast-forwarding 是"经验性"缩减（论文承认更大 LSQ 更慢、未来优化可缩短），但其正确性由断言保证，故结果仍是完整验证而非采样。

涉及论文标题：
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
