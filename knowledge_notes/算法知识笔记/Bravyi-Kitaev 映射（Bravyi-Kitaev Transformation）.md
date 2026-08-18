## Bravyi-Kitaev 映射（Bravyi-Kitaev Transformation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Bravyi-Kitaev（BK）映射是把费米子（fermion）哈密顿量编码到量子比特的方法之一，与 Jordan-Wigner（JW）、parity 变换并列。JW 把每个费米子轨道占用数局域存在单个 qubit（代价是产生/湮灭算子的 Pauli 权重 O(n)，随系统线性增长）；BK 用递归编码（Update/Flip/Parity/Remainder 四类 qubit 集合，通过递归矩阵 β_n/β_n^{-1} 在占用数基与 BK 基之间变换）同时非局域存储占用数与 parity，使单个费米子算子的 Pauli 权重降到 O(log n)（单产生/湮灭从 O(n)→O(log n) qubit 操作），适合大系统电子结构模拟。本论文所有费米子模型（Fermi-Hubbard、LiH、HF、PD-1 蛋白）都用 BK 映射转成自旋哈密顿量后输入编译流水线。
- 相关变体：Bravyi-Kitaev Superfast（BKSF，Setia & Whitfield 2018）、BK-tree、symmetry-conserving 变体，OpenFermion 提供 bravyi_kitaev() 等实现；Qiskit Nature 提供 BravyiKitaevSuperFastMapper。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pipeline 位置：物理系统 → 二次量子化（费米子算子）→ BK 映射 → 自旋/Pauli 哈密顿量 → 本论文的 Kernpiler 编译（输入即 BK 映射后的 Pauli 串集合）。
- 编码逻辑（BK）：对每个轨道 i 维护 qubit 集合 U(i)（占用数变化时被翻转）、F(i)（parity 对应子集）、P(i)（低索引轨道 parity）、R(i)（F(i) 在 P(i) 内的补集）；产生算子 a_i^† 的 Pauli 表示为 (X_{U(i)} ⊗ ...) 的产物，权重 O(log n)。论文在可扩展性分析中特别指出：BK 映射下 Pauli 项权重随系统对数增长，因此大系统时权重项可能超出 partition 大小（n=3/5）装不下，导致非对易对占比呈对数曲线（图 9）——可用 constant-weight 映射（Derby et al. 2021）缓解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：OpenFermion 的 opconversions 模块（bravyi_kitaev / bravyi_kitaev_fast / bravyi_kitaev_tree）、PennyLane qml.bravyi_kitaev、Qiskit Nature 的 mapper 类。使用时：给定费米子哈密顿量（如 Fermi-Hubbard：qubit#=2×site，本论文 8-128 qubits；分子：LiH/HF 10 qubits；PD-1 蛋白 28-222 qubits），BK 映射输出自旋哈密顿量 Pauli 串集合，作为 Kernpiler 的输入。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
