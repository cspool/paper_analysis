## Qiskit PauliEvolutionGate（含 Paulihedral 与 Rustiq 集成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PauliEvolutionGate 是 Qiskit 中把哈密顿量时间演化算子 e^{iHt}（H 为 SparsePauliOp）转化为量子线路的标准门/插件：内部先做 Trotter 化（默认一阶 Lie-Trotter），再把每个 Pauli 项指数化为 CNOT ladder + RZ 线路，并支持通过 HLSConfig/HighLevelSynthesis 插件（如 PauliEvolution.rustiq）替换底层合成算法。本论文用它作为两个 state-of-the-art baseline 的载体：Qiskit PauliEvolutionGate（默认，底层为 Paulihedral 风格的同时对角化+重排）与 Rustiq 版 PauliEvolutionGate（Pauli network synthesis）。评估用 Qiskit stable 1.3.2。
- Paulihedral（arXiv:2109.03371，ASPLOS 2022）：块式（block-wise）编译框架，用 Pauli IR 在 Pauli 串层表示线路，利用互对易 Pauli 串可同时对角化（共享 Clifford 基变换）的性质把对易项分块合成，显著降门数与深度（vs t|ket⟩ 超导后端平均 -53.1% 门数/-53.3% 深度）。Rustiq（arXiv:2404.03280）：Rust 实现、更快更短的 Hamiltonian simulation 线路合成（Pauli network synthesis），集成进 Qiskit transpiler pass（PauliEvolutionSynthesisRustiq）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 作为 baseline 在评估中的流程（论文 5 节）：输入同 Kernpiler 的哈密顿量 → 用 Qiskit 生成线路（一阶 Lie-Trotter / 二阶 Trotter-Suzuki，仅改 steps 参数）→ Qiskit transpiler level 3、u3+CNOT basis、all-to-all 优化 → 数值转格式算 L2 范数 → 与 Kernpiler 对比 depth/CX/U3。Rustiq 版在默认版基础上替换为 rustiq synthesis（PauliEvolution.rustiq 插件）。
- 表现特征（图 6 讨论）：PauliEvolutionGate 对"规则、低连通、低权重"哈密顿量最好（对称局域连接，如规则自旋模型）；Rustiq 对"分子/电子结构、非平凡连通与项"最优（by design）。Kernpiler 相对两者中较优者：depth/CNOT 最多减 86%（平均 40%）、单比特门最多减 85%（平均 11%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：`PauliEvolutionGate(SparsePauliOp, time)` + transpile；插件用法 `HLSConfig(pauli_evolution=[("rustiq", {...})])` 经 HighLevelSynthesis 调用。Rustiq 源码 https://github.com/smartiel/rustiq-core（对应论文 arXiv:2404.03280）；Paulihedral 论文 arXiv:2109.03371。使用时：本论文把 PauliEvolutionGate 作为"门级优化 baseline"代表（不含误差缩减优化），Rustiq 版作为"当前最优综合"代表，共同作为 Kernpiler 对比对象。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
