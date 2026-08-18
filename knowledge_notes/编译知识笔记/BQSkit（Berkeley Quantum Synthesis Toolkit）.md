## BQSkit（Berkeley Quantum Synthesis Toolkit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BQSkit（Berkeley Quantum Synthesis Toolkit，读作 "bis-kit"）是 LBNL 开发的便携量子编译框架，核心是数值合成（numerical synthesis）：用参数化线路 IR（UnitaryBuilder）+ 数值优化（instantiation）把目标酉矩阵逼近为基本门线路。合成算法分自上而下结构化（QSD、KAK，2-qubit 深度最优）与自下而上数值搜索（QSearch：A*+数值优化，3-4 qubit 最优；LEAP：prefixing 扩展到 4-6；QFAST：6-7；QPredict：6-12+；QFactor：12+ qubit 分 panel）。所有合成是近似的（默认 Hilbert-Schmidt 距离 <1e-10 验证）。论文用 BQSkit 1.2.0 作为 unitary synthesis baseline。
- 与 Kernpiler 的对比定位：BQSkit/Qiskit synthesis 在"最低层表示"上做合成、无法利用输入酉的高层哈密顿量结构（无法设目标如"10 门解 vs 5 门解"），因此搜索空间大、门数不优或编译慢；Kernpiler 的 MCTS 用高层结构信息缩小搜索空间。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 作为消融 baseline（图 5）：把 Kernpiler 的 MCTS 重写替换为 BQSkit synthesis，其余流水线不变，比较 depth/CX/U3/编译时间。结果：MCTS 相比 BQSkit 各 benchmark 次优方案 U3 减 60%(峰值)/33%(平均)、CNOT 减 67%/34%、depth 减 60%/32%，且编译时间接近（BQSkit 有 cache 优化）。原因：两者都是 search-based（会尝试大搜索空间中的不同分解），而 MCTS 有高层结构信息 + 可设目标。
- 使用流程（BQSkit 通用）：`CompilationTask.synthesize(unitary)` → 内部选择合成算法（按 qubit 数）→ 数值 instantiation 优化参数 → 输出线路；验证 `dist < 1e-10`。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：`pip install bqskit`（Python 3.8+，Rust 组件用 OpenBLAS/NLOPT/Ceres）；GitHub BQSKit 组织（含 BQSKit/qsearch）。使用时：直接合成 unitary 或作为编译 pass（SynthesisPass）集成；本论文用它做 8×8（3-qubit）unitary 合成的对比工具。局限：无高层结构信息、数值合成对大规模 unitary 慢。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
