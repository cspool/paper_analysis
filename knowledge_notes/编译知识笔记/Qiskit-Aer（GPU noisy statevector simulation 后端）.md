## Qiskit-Aer（GPU noisy statevector simulation 后端）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Qiskit-Aer（github.com/Qiskit/qiskit-aer）是 IBM Qiskit 的高性能模拟器包：提供 StatevectorSimulator（态矢量，GPU 支持）、DensityMatrixSimulator（密度矩阵）、QasmSimulator（带噪声的电路级模拟）等后端，GPU 后端底层为 NVIDIA cuStateVec。本论文用 Qiskit v2.1.0（StatevectorSimulator，GPU，后端 cuStateVec v1.12.0）作为 TUSQ 的 noisy simulation baseline：对每个 shot 的电路实例顺序执行完整 SVS。TUSQ 相对 Qiskit 平均/最大加速 59.06×/7878.03×（198 benchmark，TUSQ 全部 benchmark 都快于 Qiskit）。
- 与已有 Qiskit 条目（PauliEvolutionGate、SABRE 等编译/转译组件）不同，本条目聚焦 Qiskit-Aer 的 noisy statevector 模拟执行路径，是 TUSQ 的 baseline 使用场景。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Qiskit-Aer noisy simulation 执行流程（baseline 对比场景）：
  ```
  from qiskit_aer import StatevectorSimulator
  from qiskit_aer.noise import NoiseModel, depolarizing_error
  # 电路 transpile → basis（单比特门 + CNOT）→ 加噪声通道
  noise_model = NoiseModel(basis_gates=['u1','u2','u3','cx'])
  noise_model.add_all_qubit_depolarizing_error(p, 1)   # DEP p=1%
  sim = StatevectorSimulator(method='statevector', device='GPU')
  result = sim.run(circuit, shots=10**6, noise_model=noise_model).result()
  # 内部：每 shot 采样噪声通道成固定噪声门电路 → cuStateVec SVS → 平均
  ```
- 对比 TUSQ：Qiskit 对 10^6 个电路实例逐次 SVS（无冗余消除）；TUSQ 用 ECM+DFTT 把同一任务压到约 820s（30-qubit Adder），Qiskit 超过 10 小时（Perlmutter 40 小时超时上限内部分 benchmark 未完成，图7 用 ∞ 柱标记）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：`pip install qiskit qiskit-aer`（https://docs.quantum.ibm.com/api/qiskit-aer/）；`StatevectorSimulator(method='statevector', device='GPU')` 启用 GPU（需 cuStateVec 环境）；噪声模型 `qiskit_aer.noise` 支持 depolarizing/amplitude-damping/phase-damping/measurement 等。本论文在 Perlmutter 以 Qiskit 2.1.0 跑 198 个 Supermarq benchmark（32k-10M shots）作 baseline；DMS 参照：DensityMatrixSimulator 内存 O(2^2n) 不可扩展（论文引 El Capitan 25-qubit 上限）。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
