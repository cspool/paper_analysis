## MQT QCEC（量子线路等价性检查器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MQT QCEC（Quantum Circuit Equivalence Checking，慕尼黑工业大学 MQT 开源套件 cda-tum/qcec，原 MQT Bench 系列之一）是量子线路形式化等价性验证工具：给定两个线路，判定其是否实现相同酉变换。方法基于决策图（Decision Diagrams，如 DD-based simulation）与替代构造（construction-based / simulation-based）等方案，可处理数十至上百 qubit 的大线路（相对 statevector 暴力对比的可扩展替代）。提供 Python 包 mqt.qcec 与命令行 qcec。
- 本文用途（Q6 正确性验证）：PhasePoly 引入 cross-block IR 与跨 block 合并，须保证中间状态合法可合成（Section III-C3 的 rank 约束 + 剪枝规则）；端到端验证用两把尺：(i) 对 <8 qubit 电路用 Qiskit 比较 unitary；(ii) 对所有电路用 MQT QCEC 形式化验证。所有基准通过；mod_adder_1024 因超出硬件限制除外。附带作用：PhasePoly 仓库把 QCEC 作为内置等价性检查器（依赖 mqt.qcec==3.2.0）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（验证一次 PhasePoly 输出）：输入原线路 C₁ 与优化线路 C₂（OpenQASM 2.0）→ ① 构造决策图/稳定子表示，逐门应用更新状态；② 对 C₁ 正推、对 C₂ 逆推，两表示在中途"汇聚"比较（construction-based），或独立模拟后比较终态（simulation-based）；③ 输出 equivalent / non-equivalent（含可判定性信息）。对 8–499 qubit 的 MCX/Adder/HWB 大电路也适用（决策图压缩线路结构，不依赖 2ⁿ 状态向量）。
- 在本文实验中的例子：全部基准（含大电路族）通过 QCEC 验证；仅 mod_adder_1024 因本地硬件限制跳过（论文明确说明）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：pip install mqt.qcec（依赖 mqt-core），CLI `qcec file1.qasm file2.qasm`，API 传入 QuantumCircuit 或文件路径，返回等价性结论（equivalent、not equivalent、unknown）；支持 OpenQASM 2.0/3.0 与多种线路表示。使用场景：编译 pass 回归验证、优化器正确性审计、交叉编译一致性检查。
- 与本文关系：QCEC 是 PhasePoly 正确性论证的关键工具（配合 Qiskit unitary 对比），支撑"cross-block 优化保持等价"的声明。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
