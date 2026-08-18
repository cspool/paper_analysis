## Pauli 乘积旋转（Pauli Product Rotation，P_θ）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Pauli 乘积旋转是 Pauli-Based Computation（PBC）的基本计算原语（Litinski "A Game of Surface Codes", Quantum 2019）：$P_\theta = \exp(-iP\theta)$，其中 P 是多 qubit Pauli 算子（X/Y/Z 的 tensor 积），θ 为旋转角。Clifford+T 门集的元素可表示为 P_θ 的特例：S=Z_{π/4}、T=Z_{π/8}，标准分解 H=Z_{π/4}X_{π/4}Z_{π/4}、CNOT=(Z⊗X)_{π/4}(I⊗X)_{−π/4}(Z⊗I)_{−π/4}。化简规则：若 Pauli 算子 P 与 P' 交换（PP'−P'P=0），P_{π/4} 可越过 P'_θ；若反对易，P'_θ 变成 (iPP')_θ。Clifford 门把 Pauli 映射到 Pauli，可吸收进最终测量。执行方式：非 Clifford 旋转（如 π/8）经 magic state teleportation 消费 |T⟩ 态实现；π/4 与 π/8 Pauli 乘积测量用标准 gate teleportation 协议（Litinski [34]）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- P_θ 在 O3LS 编译流水线中的位置：QASM → ①Clifford+T 分解（GridSynth，容限 10⁻⁵）→ ②Pauli-based transpilation 生成 PPR 序列（P_θ 及其化简）→ ③表面码映射与调度。执行 P_θ 的具体过程（以 π/8 旋转为例）：目标 Pauli 串（如 Z_0Z_1Z_2）→ 初始化 ancilla patch → 与 magic state |T⟩ 做 gate teleportation（π/4 与 π/8 PPM 按 [34] 协议）→ 若测量结果为 1 则补条件校正 exp(iπ/4·Z_0Z_1Z_2)。Y 算子（Y^{⊗N} 旋转）因 X/Z 不能同时访问需先经 Y-synthesis 分解为 X/Z 组合（偶数个 Y 时二分分组、吸收抵消）。示例：$(Y^{\otimes N})_{\pi/8}$ 分解为 $[(Z^{\otimes n})_{\pi/4}\otimes(Z^{\otimes N-n})_{\pi/4}](X^{\otimes N})_{-\pi/8}[(Z^{\otimes n})_{-\pi/4}\otimes(Z^{\otimes N-n})_{-\pi/4}]$（n 与 N−n 均为奇数），选可抵消的分组。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器产物（PPR 序列）与物理协议两层：编译侧用 Pauli DAG（O3LS-IR）表达依赖与并行、用交换/反对易规则化简；物理侧以 lattice surgery PPM + magic state teleportation 执行，旋转角决定是否消费 magic state。评估：STIM（d=9、p=10⁻³）表征 PPM/PR/measurement 错误率，分层 LER 模型累加。工具：Qiskit LitinskiTransformation、PennyLane Pauli Product Rotations 编译插件（cite 论文）实现 PBC 到 PPR/PPM 的转译。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
