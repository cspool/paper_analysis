## CNOT 网络合成与 GF(2) 线性可逆电路合成（CNOT Network Synthesis / Gaussian Elimination over GF(2)）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 线性可逆电路合成：给定 GF(2) 上可逆方阵 A，求最小 CNOT 序列实现该线性变换（每个 CNOT 对应一次初等行操作 row_j ← row_j ⊕ row_i）。经典算法：高斯消元——把 A 逐步归约为单位阵，每步记录对应 CNOT，逆序即合成网络；复杂度多项式，但门数非最小（CNOT 网络合成对一般矩阵是 NP-hard，Patel-Markov-Hayes 2008 证明任意 n×n 线性变换最坏需 Θ(n²/log n) 个 CNOT，最优合成 NP-hard）。常见实用方法：高斯消元（O(n³)）、Patel 算法（下三角/上三角分解，最坏 O(n²/log n)）、贪心 Gray 码合成（Gray-Synth 用于相位项排序）。
- 在相位多项式优化中它承担"实现 output-parity 网络"与"实现 phase-parity 网络"两个角色：CNOT 既是构造 phase parity 的机制（把 parity 传播/对齐到可发射旋转的 qubit），也是实现输出基变换 g 的机制。
- 论文要点：单个 CNOT 对应 GF(2) 行操作，CNOT 线路 ⟺ 从单位阵出发的矩阵更新序列；合成 CNOT 网络 ⟺ 把矩阵归约回单位阵（Fig.6）。phase-parity 矩阵因非方阵不能直接高斯消元，需与 output 矩阵耦合处理（见耦合矩阵条目）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 高斯消元合成伪代码（output-parity）：输入 g（n×n 可逆矩阵）→ for col=0..n-1：找到 pivot 行（第 col 列为 1 的行 r）→ 对每行 k≠r 且 g[k][col]=1：CNOT(r, k)（row_k ← row_k ⊕ row_r）→ 直到矩阵变单位阵 → 记录的全部 CNOT（逆序）即合成网络。张量例子（Fig.6）：G₁ → CNOT(q₀,q₁) 使 row(q₁) = [1,1,0,0]；合成即逆向归约。
- phase-parity 侧计算过程（本文耦合矩阵）：CNOT(i,j) 的约定是"更新 phase-parity 项而非量子态本身"（沿用 Amy 2018）：row_i ← row_i ⊕ row_j 同时作用于 phase 与 output 两 block；当 phase 列 Hamming weight 降到 1 时，对应 parity 只依赖单 qubit，可立即发射 Rz 并删列（Fig.7 例：CNOT(q₁,q₀) 使 (110)ᵀ→(100)ᵀ）。重复直至 phase 列清空，剩余 output 矩阵高斯消元。
- NP-hard 的应对：PhasePoly 用 space-bounded A* 搜索（priority queue 上限 + 多解池 k，f=g+h₁+h₂）而非精确最优；搜索空间按"active row pair"（能降低 active column set 中某列 Hamming weight 的行对）剪枝避免 livelock。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：经典侧线性代数库或位运算实现 GF(2) 矩阵操作。开源参考：PhasePoly（https://github.com/ruadapt/PhasePoly，row_heap 合成器）、T-par/Gray-Synth（Amy 系列，arXiv:1804.06022）、Qiskit 的 LinearFunction 合成（Patel 算法）、BQSkit（Berkeley Quantum Synthesis Toolkit）等。使用场景：量子线路逻辑优化（output 基变换合成）、Clifford 电路化简、量子编译器中线性映射到拓扑的预优化。
- 与本文关系：PhasePoly 把"output 网络事后高斯消元"升级为"与 phase 网络联合的 A* 搜索"，并证明在耦合视角下整体 CNOT 更优（优于分开处理 9.21 个百分点 vs Gray-Synth）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
