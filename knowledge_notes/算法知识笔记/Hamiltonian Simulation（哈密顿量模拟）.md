## Hamiltonian Simulation（哈密顿量模拟）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hamiltonian simulation 指按照目标量子系统的哈密顿量（Hamiltonian，系统的总能量算子 H）去演化一组量子比特：给定初始态 |ψ(0)>，求 t 时刻的态 |ψ(t)> = e^{iHt}|ψ(0)>（含 iH 约定），其中 e^{iHt} 是时间演化算子。它是量子计算的核心价值场景之一（Feynman 原始动机），广泛应用于材料科学、量子化学（分子能谱）、核物理与高能物理（格点场论）、凝聚态（Fermi-Hubbard、Heisenberg、Ising 模型）等经典方法难以处理的系统。困难在于：直接实现 e^{iHt} 需要把 H 分解为可执行的量子门序列，而对一般 H 无闭式分解，必须用近似方法（product formula / Trotterization、LCU、qubitization 等）。本论文目标即为哈密顿量模拟的编译优化：在达到给定精度（L2 范数 >99.5%）下最小化门数与电路深度。
- 关键组成部分：H 的 Pauli 串分解（H=Σ_i w_i P_i）、时间演化算子的近似展开（Trotter 乘积公式）、把近似展开编译为基本门线路（unitary synthesis），以及误差度量（L2 范数差矩阵、保真度 ≈1−(L2 norm)²）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 常规算法 pipeline：输入物理模型 → 二次量子化（费米子哈密顿量）→ 费米子到量子比特映射（Jordan-Wigner / Bravyi-Kitaev，见 Bravyi-Kitaev 条目）→ 把 H 分解为加权 Pauli 串（如 H=JΣ⟨i,j⟩Z_iZ_j+hΣ_iX_i 的 Ising 模型）→ Trotter 化展开 e^{iHt}≈(Π_k e^{iH_k t/N})^N → 每个 Pauli 指数 e^{iP t} 用 CNOT ladder + RZ 门实现 → 门级优化（重排、同时对角化、synthesis）→ 输出线路在量子硬件/模拟器上执行并测量。
- 本论文示例（1D Ising，图 3）：输入为部分 Trotter 化的 unitary 集合 {e^{i(ΣH_i)t}}，共同组成一个 Trotter step；编译输出为可执行线路。仿真验证用 8-10 qubit 的 LiH/HF 分子、Ising/Heisenberg/Fermi-Hubbard 自旋模型，扩展性到 28-220 qubit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：软件生态包括 Qiskit（PauliEvolutionGate，含 Rustiq/Paulihedral 集成）、Cirq、OpenFermion（分子哈密顿量生成与 fermion-to-qubit 映射）、BQSkit（unitary synthesis）。物理实现依赖量子硬件（超导、离子阱等）或经典 statevector 数值仿真（本论文在 A100 GPU + EPYC CPU 上做数值验证，未用真实量子硬件）。使用时：用户给定 H 与时间 t，编译器输出 e^{iHt} 的近似线路；精度由 L2 范数差矩阵度量，目标保真度决定所需 Trotter 步数与线路规模。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
