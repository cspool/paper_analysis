## Supermarq Benchmark Suite

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Supermarq（Tomesh et al., HPCA 2022，"SupermarQ: A Scalable Quantum Benchmark Suite"，github.com/PrincetonQuantum/Supermarq）是可扩展量子线路基准套件，用于评估量子软件栈（编译器、模拟器、噪声处理等）。TUSQ 用它作为 noisy statevector 模拟的基准集：选择 QAOA、Adder、Bitcode、Phasecode、GHZ、QFT、BV 七类电路，覆盖多种结构（线性：GHZ/Bitcode/Phasecode；并行：QAOA）、qubit 数（13-28）、深度（4-770）、门数（4-1250）与输出分布形态（单峰 Adder/Bitcode/Phasecode、双峰 GHZ、尖峰 QAOA、均匀 QFT）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TUSQ 的 benchmark 配置（表 I）：QAOA 13-25 qubit、depth 82-770、130-1250 门；Adder 4-28、69-289、97-417；Bitcode 5-25、4-144、4-144；Phasecode 5-25、8-48、20-470；GHZ 14-28、14-28、14-28；QFT 14-24、27-47、105-300；BV 4-24、6-26、14-74。评估流程：每个电路加噪声模型（DEP/measurement/Pauli-twirled damping，p=1%）→ 设定 shots（32k/100k/1M/10M）→ 分别跑 TUSQ 与 Qiskit 2.1.0/CUDA-Q 0.11.0/TQSim → 算 speedup γ 与 relative fidelity difference δ。
- 趋势：speedup 随 qubit 数增加（模拟时间指数增长放大任何加速）；固定 qubit 时 speedup 随 QAOA 层数 p 增加而下降（深度增加使 ER 更多样、Tallying/Pruning 有效性降低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 使用：pip install supermarq 或从 GitHub 拉取，生成标准电路（如 `from supermarq.benchmarks.qaoa import QAOA`）并施加噪声再交给模拟器。TUSQ 用它评估 198 个 benchmark（含不同 qubit/深度/shots 组合）并报告平均/最大加速 59.06×/7878.03×（vs Qiskit）与 13.38×/439.38×（vs CUDA-Q）。VQE 正确性验证用 Ising/Heisenberg Hamiltonian（10/15 qubit）而非 Supermarq。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation
