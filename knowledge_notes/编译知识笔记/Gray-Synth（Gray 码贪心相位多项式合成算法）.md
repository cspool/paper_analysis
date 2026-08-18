## Gray-Synth（Gray 码贪心相位多项式合成算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Gray-Synth 是 Amy、Azimzadeh 与 Mosca 于 2018 年提出的相位多项式 CNOT 网络贪心合成算法（"On the CNOT-complexity of CNOT-PHASE circuits"，arXiv:1804.06022，QIP 2018 最佳论文之一）。输入一个 CNOT-PHASE（{CNOT,Rz}）电路的相位项集合，输出实现该相位函数的 CNOT 网络：贪心地按 Gray 码（相邻码字仅 1 bit 翻转）顺序排序相位项——从全零 parity 出发，每步只翻转一位使连续两个相位项共享的 qubit 差异最小，相邻 parity 只需一个 CNOT 即可变换，从而复用中间 parity、最小化 CNOT 数。建立在 Amy 2013 的 T-par（sum-over-paths 相位多项式 T-count 最小化）之上。
- 局限性（本文 Motivation）：理论仅适用于单 block、且只优化 phase-parity（相位旋转）部分；不系统处理相位项 XOR 传播与 block 输出基变换（g 函数）的交互；不能处理含非相位多项式门（H）的一般线路。因此它作为单 block phase-only baseline 出现。
- 本文数据：Gray-Synth 报告平均 CNOT 减 17.62%；PhasePoly 平均 26.83%（改善 9.21 个百分点）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一次合成）：输入相位项集合（parity + 角度），如 {q₀:π/4, q₀⊕q₁:π/2, q₁⊕q₂:π/4} → ① 按 Gray 码构造 parity 序列：0…00 → … 使相邻 parity 仅一位不同（如 000 → 100 → 110 → 111 → 011 → 001 中取含所需项的路径）；② 从全零态出发，对每步差异位做 CNOT，得到该 parity；③ 在该 qubit 上施加对应 Rz(θ)；④ 沿序列移动，相邻 parity 共享 CNOT，累计 CNOT 数≈序列长度。输出 = 实现相位函数的 CNOT+Rz 网络；之后输出基变换 g 由独立的高斯消元合成（两段式，无联合优化）。
- 与 PhasePoly 对比：PhasePoly 的 row_heap（A*）在耦合矩阵中联合优化 phase+output，并支持跨 block；Gray-Synth 作为 single-block、phase-only 的对照基线（论文 Q1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：T-par/Gray-Synth 的 Python/C 实现见 Amy 主页与 arXiv:1804.06022 配套代码；PhasePoly 仓库（https://github.com/ruadapt/PhasePoly）内置 single_block_greedy 等作为可对比合成器；Qiskit 也有 phase polynomial synthesis 相关 pass。使用场景：{CNOT,Rz} 单 block 逻辑优化、CNOT 网络合成研究、作为更复杂优化器的子步骤或基线。
- 与本文关系：论文将其作为相位多项式基线之一，并指出其"单 block + phase-only"限制正是 PhasePoly 联合优化与跨 block 表示要突破的。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
