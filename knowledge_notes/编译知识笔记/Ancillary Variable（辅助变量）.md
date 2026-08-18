## Ancillary Variable（辅助变量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ancillary variable（辅助变量/辅助位）是 SAT-to-QUBO/Ising 公式化过程中引入的不对应原 SAT 问题变量的额外二值变量，用于把子句约束"编码"进二次能量函数。以 3SAT 子句 C=(x1∨x2∨x3) 的 Chancellor's 公式化（Eq.(3)）为例：ancillary a1 通过项 −a1(x1+x2+x3−2) 参与能量，让子句满足时存在 a1 使能量取最低值（表 I：满足行 i=1..7 时 min(H_C1)=−1），而子句不满足（i=0）时 H=0 为局部高能。ancillary 的价值：把逻辑约束转成"二次多项式能量最小化"目标，使 Ising 机/退火器可求解；代价：ancillary 增加 QUBO/Ising 实例规模，直接缩小可装入硬件的 SAT 问题尺寸（n+m spin for 3SAT），且其取值原则上不应被固定（否则破坏子句能量语义）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC 编译框架中，ancillary 是正确性问题的核心：编译必须满足 ancillary-awareness——形成子问题时每个 clause 的 ancillary 必须与其问题变量同组，否则被排除的 ancillary 只能被 solver 定成固定值，导致该子句能量景观与原始目标错位。关键设计：SATIC 在 CNF 层（尚无 ancillary）做子问题形成，按构造避免遗漏；freeze 未选变量 + 单元传播把 3SAT 子句降为 2SAT/1SAT（无需 ancillary），从而"公式化后"ancillary 更少、QUBO 更小。流程示例：F=C1∧C2 冻结 x4=0、x5=0 后子 CNF (x1∨x2∨x3)∧(¬x3)，Chancellor's 仅引入 1 个 ancillary（而非 2 个），4 spin 恰好装满 4-spin 机器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：公式化例程按子句 k 自动生成 ancillary 及其二次项系数（Chancellor's 每 3SAT 子句 1 个、ILP 每 kSAT 子句 log₂(k) 个、Flat ILP 摊平为多个低权 ancillary）。使用：Ancillary Estimation 按子句宽度 k 查表预估算 ancillary 数（Chancellor's 约 2k−5 每 kSAT 子句）以在形成子问题时快速判断 QUBO 规模；D-Wave EID 等 QUBO 层分解器因不感知 ancillary 而最易在此出错（受控实验：违反 ancillary-awareness 时 UF75 仅解 44/100，D-Wave EID 在 UF20 以上即失败；D-Wave EID++ 随机化 ancillary 值只能部分缓解）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
