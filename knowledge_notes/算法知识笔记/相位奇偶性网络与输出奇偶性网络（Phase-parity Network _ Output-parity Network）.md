## 相位奇偶性网络与输出奇偶性网络（Phase-parity Network / Output-parity Network）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 在 sum-over-paths 表示 U|x⟩ = e^{i·p(x)}|g(x)⟩ 中，相位多项式优化天然分为两个子网络：(1) phase-parity 网络——用 CNOT 构造各个相位项所需的输入 XOR parity（phase-parity，如 q₀⊕q₁），并在对应 qubit 线上施加 Rz(θ) 旋转；每个相位项对应一个 parity 列向量（如 (110)ᵀ 表 q₀⊕q₁），全部相位项构成非方阵的 phase-parity 矩阵（列=parity 项，行=qubit）。(2) output-parity 网络——实现输出基变换 g(x) 的 CNOT 网络，g 是 GF(2) 上的 n×n 可逆方阵，每行/每列编码一个输出 parity（如 g(q)=(q₀, q₀⊕q₂, q₀⊕q₁⊕q₂)）。
- 关键区别：phase-parity 矩阵不可用高斯消元规约到单位阵（非方阵，列代表"要实现的旋转条件"而非"输出映射"）；output-parity 矩阵可经高斯消元规约到单位阵（CNOT 网络合成经典问题）。二者共享同一组 CNOT 行操作，因此是耦合的。
- 论文核心洞察：先前工作把两个网络分开优化（phase 网络用贪心/phase-only，output 网络事后高斯消元），但同一 phase 实现可诱导不同 output 代价——本文 Fig.3 给出两个均以 2 CNOT 最小实现同一 phase 函数的电路，其 g 函数代价分别为 2 与 3 CNOT。因此分开处理会错过联合最优。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 联合优化张量计算（本文 Fig.7）：把两者统一为耦合 parity 矩阵 [phase-parity | output-parity]。例：3-qubit 电路，phase 项 (110)ᵀ、(011)ᵀ，输出 g(q)=(q₀,q₀⊕q₂,q₀⊕q₁⊕q₂) 转置后列向量。初始 M=[[1,0,1,1,1],[1,1,0,0,1],[0,1,0,1,1]]（前 2 列 phase，后 3 列 output）。CNOT(q₁,q₀) 令 row₀ ← row₀ ⊕ row₁ → [[1,0,1,1,1],[0,1,1,1,0],[0,1,0,1,1]]，phase 列 (110)ᵀ 变 (100)ᵀ（Hamming weight 1）→ 该 Rz 可发射、删列；继续行操作清空 phase 列后，剩余 output 矩阵用高斯消元合成。伪代码：while 存在 phase 列：选 active row pair (i,j)（能降低某列 Hamming weight）做 row_i ← row_i ⊕ row_j；若某 phase 列变单位向量 → 发射 Rz(θ) on 对应 qubit 并删列；最终对 output 矩阵做高斯消元。A* 搜索以 f=g+h₁+h₂（已用 CNOT 数 + phase 矩阵总 Hamming weight + output 高斯消元估计）引导选择。
- 作用：把"先 phase 后 output 的两段式贪心"改为"同一搜索空间内联合最小化"，找到整体 CNOT 最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：以 GF(2) 矩阵（numpy/位运算）表示 parity；CNOT 即行 XOR。基线实现：Single-block Greedy（phase 用贪心、output 用高斯消元，本文复现平均总门减 26.93%/两比特门减 8.14%）；Gray-Synth 只优化 phase 部分（平均 CNOT 减 17.62%）。PhasePoly 的 row_heap 合成器用 space-bounded A* + 多解池实现联合优化（平均总门减 34.70%、CNOT 减 26.83%）。开源：PhasePoly（https://github.com/ruadapt/PhasePoly）内置 row_heap / single_block_greedy 等 6 种合成方法可对比。
- 使用场景：任何以 {CNOT,Rz} 为主的线路（算术、MCX、QAOA、Hamiltonian 模拟、FT 基准）的逻辑优化；也用于硬件感知 CNOT 合成（Parity 网络可映射到特定拓扑）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
