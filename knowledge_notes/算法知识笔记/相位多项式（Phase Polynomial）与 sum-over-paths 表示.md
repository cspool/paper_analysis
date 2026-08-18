## 相位多项式（Phase Polynomial）与 sum-over-paths 表示

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 相位多项式是量子线路中一类 {CNOT, Rz} 子电路（phase polynomial circuit）的紧凑代数中间表示（IR），由 Amy、Maslov 与 Mosca 在 2013 年以 sum-over-paths（路径求和）形式引入（"Polynomial-Time T-Depth Optimization of Clifford+T Circuits via Matroid Partitioning" 等系列工作，T-par 工具）。其核心形式（本文 Eq.1/2）：对任意计算基态 |x⟩（x∈F₂ⁿ），电路作用可写为 U|x⟩ = e^{i·p(x)} |g(x)⟩，其中 p(x) = Σ_i θ_i·(x₁y_{i1} ⊕ … ⊕ xₙy_{in}) 是布尔 parity 的带权（旋转角 θ_i）线性组合（相位函数），g(x) 是 CNOT 网络实现的 GF(2) 线性可逆变换（输出基变换）。换言之：CNOT 计算输入变量的 XOR parity（奇偶性），Rz(θ) 在这些 parity 上施加相位旋转。一个 phase polynomial block 是通用线路中只含 {CNOT, Rz} 的极大连续子电路；H 等换基门会终止该 region。
- 该表示的精妙处：把"门序列"抽象为"一组 (parity 向量, 旋转角) 对 + 一个线性变换矩阵"，从而把线路优化转成矩阵/线性代数问题（parity 项合并、CNOT 网络合成），并可精确用于等价性验证（两个 {CNOT,Rz} 电路相位多项式相同则酉等价）与硬件感知合成。Rz 包含 Clifford 旋转（Z、S）与非 Clifford 旋转（T），因此 {CNOT,Rz} 同时覆盖 Clifford 与非 Clifford 门，是 FT 与非 FT 编译的共同目标。
- 论文动机数据：在 MCX、Grover、Shor、QAOA、Hamiltonian 模拟等基准中，>75% 的门属于 {CNOT,Rz} 区域（部分 >90%），且 FT 视角下 CNOT 与 T 占比相当——因此相位多项式优化直接命中线路主导成本结构。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 算法 pipeline（构造相位多项式）伪代码：输入 {CNOT,Rz} 线路 → ① 初始化输出基变换 g = 单位阵（n×n GF(2)），相位项表 P = {}；② 逐门处理：CNOT(c,t) 使 parity 传播——对每个已记录相位项，若其 parity 含 qubit c 则把 qubit c 替换为 c⊕t（等价于更新 phase-parity 矩阵列）；同时 g 的第 t 行 ← 第 t 行 ⊕ 第 c 行；③ Rz(θ) on q：把当前 q 上的 parity 向量 y 加入 P（角度累加，θ₁y + θ₂y → (θ₁+θ₂)y）；④ 输出 P（相位项集合）与 g（线性变换矩阵）。张量计算例子（本文 Fig.2）：3-qubit 电路 p(q₀,q₁,q₂) = (π/4)q₀ + (π/2)(q₀⊕q₁) + (π/4)(q₁⊕q₂) + (π/4)q₀ = (π/2)(q₀⊕q₁) + (π/4)(q₁⊕q₂)（两个同名 parity 合并），g(q) = (q₀, q₀⊕q₂, q₀⊕q₁⊕q₂)；原电路 5 CNOT + 3 T，等价电路 4 CNOT + 1 T。
- 作用：把线路优化从"逐门重写"提升为"代数化简 + 网络合成"两层问题——先合并相位项（rotation merging），再最小化实现这些 parity 与输出基变换的 CNOT 网络（NP-hard，需启发式）。PhasePoly 论文在此基础上新增：phase 与 output parity 联合优化（耦合矩阵）与跨 block 表示。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：经典侧 Python/C++ 库解析 OpenQASM 线路构造相位多项式。代表性工具/算法：T-par（Amy 2013，T-count 最小化）、Gray-Synth（Amy 2018，CNOT 网络贪心 Gray 码合成）、Rotations/Quilc 与 TKET 的 rotation merging、QUESO 用多项式恒等式过滤生成等价电路类。开源：T-par/Gray-Synth 见 Amy 主页与 arXiv:1804.06022；QUESO（PLDI'23）仓库 qqq-wisc/queso；PhasePoly 开源 https://github.com/ruadapt/PhasePoly。使用场景：量子编译流水线中的逻辑优化 pass、等价性检查（phase polynomial 作为规范形）、硬件感知 CNOT 网络合成。
- 与本文关系：本文把"相位多项式仅作局部重写辅助"提升为"一等编译阶段"，并扩展到跨 block（见 Cross-block IR 条目）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
