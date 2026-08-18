## Quantum Unitary / Circuit Synthesis（量子酉合成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量子酉合成（unitary/circuit synthesis）是把一个给定的酉矩阵（如 e^{itΣH_i}，或任意 2^n×2^n 酉算子）分解为基本量子门（如 CNOT + 单比特旋转）线路的问题。分两类：(1) 解析/结构化分解——Quantum Shannon Decomposition（QSD，Shende 2006）、KAK 分解、block-ZXZ 分解（Krol & Al-Ars 2024，任意 3-qubit unitary 最多 19 个 CNOT）等，确定性但门数高；(2) 数值/搜索合成——QSearch（A*+数值优化，~3-4 qubit 最优）、QFAST、LEAP、BQSkit 的 instantiation 系列，用数值优化实例化参数化线路逼近目标，门数低但精度近似。本论文的 MCTS 重写属于数值/搜索合成，但只搜索 CNOT 骨架、单比特门用 Gauss-Newton 可微求解，并利用高层哈密顿量结构信息缩小搜索空间。
- 对比：通用 unitary 合成（QSD 等）不考虑输入酉的哈密顿量来源，门数高会抵消 partition 的收益（Challenge 2）；本论文论证用"高层 Hamiltonian 结构 + 学习算法"能获得更高效线路。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Kernpiler 中：每个 partition 的 unitary U_i（8×8）作为合成目标，MCTS 输出 CNOT 骨架 + 参数化单比特门，再由 Qiskit transpiler level 3 折叠为 (u3,cx) 门集最优线路。消融实验（图 5）把 MCTS 替换为 Qiskit unitary synthesis 与 BQSkit synthesis 验证其优势。
- 误差度量（Eq.5）：E(x) = argmin_θ || Π_{i=1}^n x_i(θ_i) − U ||_2，θ 为单比特门参数；合成质量 = L2 范数差矩阵（仿真用 statevector 数值计算，再平方估计保真度）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现生态：Qiskit（unitary synthesis / synth 工具、Quantum Shannon 相关 pass）、BQSkit（QSearch/QFAST/LEAP/QPredict/QFactor，数值 instantiation 为核心，见 BQSkit 条目）、QFAST（Berkeley）、Qiskit 的 QSD/block-ZXZ。使用时：给定目标酉矩阵（或分解目标），选择合成算法（结构化 vs 数值），输出 CNOT+单比特门线路。本论文选 MCTS 的原因：数值合成的奖励函数可设目标（门数/误差权衡）、可用高层结构缩小搜索空间，且其 CNOT 骨架在搜索中被重复发现（结论中提出可启发式/图算法替代 RL 的方向）。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
