## PhasePoly 相位多项式联合优化（耦合 parity 矩阵 + space-bounded A* 搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PhasePoly 是 ISCA 2026 论文 "Leveraging Phase Polynomials for Quantum Circuit Optimization"（Rutgers Blueprint 组，arXiv:2506.20624）提出的量子线路编译器优化 pass：把相位多项式优化从"局部重写框架的辅助工具"提升为通用编译流水线的一等阶段。核心机制：(1) 耦合 parity 矩阵 [phase-parity | output-parity]——把相位项（parity 向量列）与输出基变换（转置后列向量）放进同一 GF(2) 矩阵，一次 CNOT 行操作同时更新两 block，联合最小化 phase 与 output 网络（解决先前"分开优化错过相关性"的缺陷，Fig.3）；(2) space-bounded A* 搜索——CNOT 网络合成 NP-hard，用代价 f(n)=g(n)+h₁(n)+h₂(n)（已用 CNOT 数 + phase 矩阵总 Hamming weight + output 矩阵高斯消元估计）在 priority queue 中展开，队列设上限（满时丢低优先级节点）、多解池 k、tie-break 按 [f,h₁,h₂,−g] 字典序；(3) active row pair 剪枝——只考虑能降低 active column set 中某列 Hamming weight 的行对，避免 livelock；(4) 配合 SSA 旋转合并与 Cross-block IR（见对应条目）。效果：总门数平均减 34.70%（最高 50.00%）、CNOT 平均减 26.83%（最高 48.57%），大电路不退化。
- 与既有框架关系：正交于 subcircuit rewriting（Quartz/QUESO），组合管线中"先 PhasePoly 后重写"收益最大（重写后接 PhasePoly 再增 ≈6–13%，反序仅 0.75–1.25%）。开源：https://github.com/ruadapt/PhasePoly（v1.0.0，Python ≥3.10，qiskit==1.1.1、mqt-core==3.2.1、mqt.qcec==3.2.0、networkx、sympy、depq、numpy），OpenQASM 2.0 输入输出，内置 Qiskit+MQT-QCEC 等价性检查；合成方法 6 种：row_heap（默认 A*）、row_heap_classical_GE、single_block_greedy(+_classical_GE)、pure_rotation_merging/rotation_merging。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架运转流程（一次 {CNOT,Rz} 电路优化）：输入 OpenQASM 2.0 → ① 预处理器：X 门经 Clifford 共轭前推、消相邻 H 对、处理 H-bracketed CNOT，使 H 成为唯一 block barrier（与旋转合并交错两轮）；② block 划分：H barrier 切出极大 {CNOT,Rz} 子电路；③ 建耦合矩阵：phase 列（parity 向量）+ output 列（g 转置），例 Fig.7 3-qubit：M=[[1,0,1,1,1],[1,1,0,0,1],[0,1,0,1,1]]；④ A* 搜索：active row pair 扩展（CNOT(i,j): row_i←row_i⊕row_j 同步两 block），phase 列 Hamming weight=1 即发射 Rz 删列，f 代价引导，priority queue 上限 + 解池 k；⑤ phase 列清空后 output 矩阵高斯消元；⑥ 输出等价线路。例子：5 CNOT+3 T → 4 CNOT+1 T（Fig.2）。作用：在"代数化简 + 网络合成"两层同时最优，而非先前"phase 贪心 + output 事后高斯消元"的两段式。
- 与 cross-block 组合时（见 Cross-block IR 条目）：post-H 行 inactive 参与 A* 候选但被锁定，pre-H 行满足 rank 检验（rank(M∪{v})=rank(M)）才消除并激活后继行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Python 包，命令行或 API 输入 OpenQASM 2.0 线路输出优化线路；参数 Q（priority queue 上限）、P（解池大小）、G（cross-block group 大小），Q=P=1000 即达近最优（继续放大仅增编译时间，20× 解池→20× 时间收益近零）。评估基线：Rotation Merging、Single-block Greedy、Gray-Synth（相位多项式基线）；Quartz（https://github.com/quantum-compiler/quartz）、QUESO（https://github.com/qqq-wisc/queso，并入 guoq）作为通用重写框架组合对象。编译预算：PhasePoly ≤3,600 s/电路（最大实例 hwb8_113 104k 门 <5,500 s），Quartz/QUESO 每电路 7,200 s。硬件：2.8 GHz AMD EPYC 7313 CPU（经典侧编译）。
- 使用场景：FT 编译（Clifford+T 合成之前运行）、近短期硬件编译（映射路由之前减两比特门，Qiskit SABRE 后物理深度减 28.35%、大电路 40.84%）、大电路逻辑优化（MCX/Adder/HWB 三族线性增长 vs 重写框架饱和）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
