## Gate Teleportation（门隐形传态）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 用预先制备的魔法态 + 只含 Clifford 门的电路实现非 Clifford 门（如 T 门）的标准机制。逻辑链：①准备 ancilla 魔法态 $|T\rangle=(|0\rangle+e^{i\pi/4}|1\rangle)/\sqrt 2$（= T|+⟩）；②目标 qubit 与 ancilla 之间做 transversal CNOT 纠缠；③测量 ancilla；④按测量结果施加条件 Clifford 校正（对 T 门是 $S^\alpha$ 门）。净效果 = 对目标施加 T 门，全程只用了可容错实现的 Clifford 操作——绕过 Eastin-Knill 定理。代价：每个 T 门消费 1 个魔法态，因此魔法态制备（蒸馏/培养）成为 FTQC 成本主体，运行时 teleportation 相对便宜。变体：1-bit teleportation 与 Knill teleportation（qubit 数、连接性、分布式场景有取舍）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文把 teleportation 实例化为 exp(iπ/8·P) 的三种注入方案（图 4）：标准注入随机产生 exp(±iπ/8·P)，需条件校正 exp(iπ/4·P)。①Direct injection + factory correction（a）：|T⟩ 经 inter-module 测量直接 teleport 到全部目标 qubit，校正用 measurement-to-rotation 电路显式实现；②Pivot injection（b，默认）：|T⟩ 先 teleport 到 pivot L0 再经 in-module 测量传到目标，校正吸收进 pivot 的条件 X/Y 测量——把噪声注入限制在单 qubit、inter-module 噪声集中在源-pivot 界面；③Direct injection + source correction（c）：源模块支持高保真 Y 测量时由源自行校正，pivot 完全不参与。错误传播细节：注入错误中 Z 分量等价于 Z 制备错误（多引入 π/2 旋转）、X 分量可被最终 X 基测量吸收，故魔法态 qubit 上的错误危害较低。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- surface-code 语境：lattice surgery 工厂蒸馏出的 |T⟩ 经 surgery 路由到消费点 teleport（Litinski 系列）；本论文在 QLDPC 语境经 adapter（universal surgery ancilla）做 inter-module 测量注入。工程上还需 byproduct Pauli 跟踪：所有条件校正都是 Clifford/Pauli，可经典跟踪延迟处理。Web 参考：PennyLane magic states 教程、Qualtran TGate 文档、arXiv:2502.16939（rotated surface code 上 T 态消耗的扩展稳定子仿真）。
- 补充（TACO 论文）：TACO 把 gate teleportation 作为 T/Rx(π/4) 门的标准实现并给出其硬件成本模型：消费已备好的 magic state 需 2.5d+4 QEC cycles（d=19 时约 51.5 cycles），S 门经 |Y⟩ 态 teleportation 共 1.5d+3 cycles；magic-state injection 符号概率性（±），若符号相反用 Clifford correction 恢复——比再做一次非 Clifford 旋转便宜得多，故可接受。teleportation 的"每门一个 magic state"属性使 TACO 把 magic state 吞吐当作架构参数（compute block 每 cycle 一个 π/4 旋转，需 4 个 magic states/cycle 配置），并由高 π/4 旋转局部性保证 target qubit 驻留 compute block 避免移动开销——teleportation 的消费端成为架构 locality 优化的直接对象。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
