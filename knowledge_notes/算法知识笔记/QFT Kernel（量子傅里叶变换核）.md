## QFT Kernel（量子傅里叶变换核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量子傅里叶变换（Quantum Fourier Transform, QFT）是 Shor 算法、量子相位估计（QPE）等核心算法的基本子程序：对 n-qubit 态施加 QFT 需要 O(n²) 个受控相位门（CPhase，每个作用于一个 qubit 对），其结构是"蝴蝶式"（butterfly）的相位阶梯。QFT 的 2Q 门全连接模式（任意 qubit 对都可能出现 CPhase）使它在受限拓扑上路由开销显著，是 qubit routing 的经典 benchmark 与"深度最优性"研究对象（Maslov 手工最优方案、TOQM 的 A* 深度最优等）。
- CANOPUS 论文把它作为第一个 case study：证明 n-qubit QFT 在 1D chain 上的最小 SWAP 插入数为 n(n−1)/2 − 2（比 CPhase 数少 2），形成完美的对称蝴蝶结构（Fig.8(b)）；CANOPUS 在 1D chain 上对所有规模达到该理论最优，超越 TOQM（声称实现 Maslov 方案却失败）与 Maslov 手工方案（多 2 个 SWAP）——该结论与目标 ISA 无关。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- QFT 核结构（n=6 示例，Table I）：qft_6 路由前 #Can=15（CPhase 数）、Depth2Q 与最优路由后同为 15/9；qft_12 为 66/21（CANOPUS 在 1D chain 上达到最优，TOQM 为 67/22）。QFT 编译 pipeline：① 生成 QFT 逻辑电路（CPhase 序列，每层一个控制 qubit）；② 逻辑级优化（TKET）+ rebase 为 {Can,U3}；③ 路由（CANOPUS/TOQM 等）→ 物理电路（#Can、Depth2Q 指标）；④ 真机执行（可选）。
- 真机验证例子（V-A）：在 IBM ibm_marrakesh（Heron-R2 QPU，native 门 {CZ, √X, Z(θ), ZZ(θ)}，heavy-hex 拓扑但含足够 1D chain）上编译并执行 n∈{6,8,10,12} 的 QFT，用 Hellinger fidelity 测量（实验 vs 理想输出分布；shots = MAX{4096, 2^n×10}；每个电路追加一层 Hadamard 使理想终态为 |0>^⊗n）。CANOPUS vs QISKIT 默认编译：CZ 门数降 52.9%、2Q 深度降 66.4%、CZ/CX 门集错误降 26.89%、ZZ(θ) 门集错误降 34.98%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：QFT 在 Qiskit 中可用 `qiskit.circuit.library.QFT` 生成；CANOPUS 仓库 `python route_qft.py <n>` 直接对比 SABRE 与 CANOPUS 的 n-qubit QFT 路由结果（#Can 与 2Q 深度）。作为 benchmark，QFT 出现在 QASMBench/MQTBench（论文 Table III 的 qft 18 qubit：#Can 153、Depth2Q 33、Ccount 306）。
- 场景意义：QFT 是"程序模式-ISA-拓扑协同"的样板——子程序展开式（subroutine-unrolling）构造的算法天然适配 chain 拓扑（heavy-hex 反而更高开销）；其 CPhase 全连接模式为 ISA-aware SWAP absorption 与交换性优化提供丰富机会，也是跨编译器（CANOPUS/TOQM/QISKIT）公平比较的黄金 benchmark。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
