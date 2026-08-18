## Pauli String（Pauli 串）与哈密顿量分解

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pauli string 是 n-qubit 系统中长度为 n 的张量积算子 P = ⊗_{i=1}^n σ_i，其中每个 σ_i ∈ {X, Y, Z, I}（单比特 Pauli 算子或恒等）。所有 Pauli 串构成 n-qubit Hermitian 算子线性空间的一组完备基，因此任何哈密顿量（Hermitian 算子）都可分解为加权 Pauli 串之和：H = Σ_i w_i P_i（w_i ∈ R，本论文把权重吸收进项记 H=Σ_i H_i）。单个 Pauli 串的指数 e^{iPt} 可用 Pauli 门 + CNOT 链 + Z 旋转门精确实现（Pauli 指数线路），但"多个 Pauli 串之和的指数" e^{itΣP_i} 一般无闭式分解，必须近似——这正是 Trotterization 与 partial Trotterization 的用武之地。
- 本论文中 Pauli 串承载全部编译输入：partitioning 排序按"最高 qubit 索引 + 权重"、conflict graph 的对易性判定（[t_i,t_j]≠0 即加边）都直接作用于 Pauli 串；term 权重与 locality 决定分组密度与误差行为。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 分解与指数化流程：
```
输入: 哈密顿量 H（如 4-qubit 自旋链）
1) 分解为 Pauli 串: H = J1*X_3 + J2*X_1X_2 + J3*Z_3Z_4 + J4*Z_1   (权重并入项)
2) 单个 Pauli 串指数: e^{iθ X_1X_2} = H_1(CNOT_{1,2} RZ(2θ) CNOT_{1,2}) H_1   (CNOT ladder + RZ)
3) 多串和指数: 用 Trotter / partial Trotter 近似
```
- 对易性判定示例：X_1X_2 与 Z_1 共享 qubit 1 且算子不同（X vs Z）→ 不对易，在 conflict graph 中连边，被分组进同一 partition（Table 1 中 {Z_1, X_1X_2}）；而 Z_1 与 Z_3Z_4 无共享 qubit → 对易。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Pauli 串在 Qiskit（SparsePauliOp/Pauli）、OpenFermion、Cirq 等 SDK 中是标准数据类型；哈密顿量分解可用 fermion-to-qubit 映射自动生成（如分子哈密顿量经 OpenFermion → Qiskit）。使用时按本论文流水线：排序（最高索引、权重）→ 贪心分组（≤3 qubit）→ 冲突图分组与重排 → MCTS 重写。局限：分解到 Pauli 串后 term 数可能很大（尤其费米子/分子哈密顿量），但 locality 与权重信息帮助排序把非对易项聚拢。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
