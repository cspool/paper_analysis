## GridSynth（Ross-Selinger z-rotation→Clifford+T 近似合成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GridSynth 实现 Ross & Selinger 的 ancilla-free Clifford+T z-rotation 近似合成算法（"Optimal ancilla-free Clifford+T approximation of z-rotations"，arXiv:1403.2975）：给定目标角 θ 与精度 ε，把任意 Rz(θ) 分解为 H、S、T 序列，T-count 在 O(log(log(1/ε))) 内接近最优（比 Solovay-Kitaev 的 O(log³(1/ε)) 指数级更好）。算法用数论（整数格基约化 LLL、Gaussian integers Z[ω] ω=e^{iπ/4}）在 Clifford+T 群中找逼近 Rz(θ) 的最短字。实现：Selinger 的 Haskell gridsynth 可执行程序、Qiskit `gridsynth_rz`（qiskit/synthesis/discrete_basis/ross_selinger.py，Rust 加速）、Quantinuum/grid_synthesis（Rust）等。
- 本文用途（Q5）：FT 编译中每个任意 Rz 必须合成到 Clifford+T 指令集（H、S、T 序列）；论文研究 PhasePoly 与 GridSynth 的两种编译顺序对最终 T 数/两比特门/深度的影响。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Q5 组合实验）：两种管线 (A) GridSynth→PhasePoly 与 (B) PhasePoly→GridSynth，最后都接相同的 commuting-rule 化简。14 个变分电路（QAOA Max-Cut 3-regular 图 4–24 qubit 2,150–12,900 门；VQE UCCSD ansatz 的 JW/BK/parity 编码 + HWPA 4–12 qubit 2,641–231,780 门）。结果：B 在多数电路上深度最低——因为 PhasePoly 先化简大 {CNOT,Rz} 区域，避免 GridSynth 引入的额外 H 门（会把 phase polynomial block 切碎、限制后续 rotation merging）。(i) T-count 变化温和（主要 HWPA 电路）；(ii) 两比特门减普遍转化为深度减；(iii) 结构化相位交互电路（parity/HWPA）平均 ~10% 深度减，JW/BK 编码 <1%（其 CNOT-Rz 结构已紧凑）。
- 意义：指导"相位多项式 pass 应放在 Clifford+T 合成之前"的编译流水线设计原则。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：GridSynth（Ross-Selinger）——Haskell gridsynth（mathstat.dal.ca/~selinger/newsynth）、Qiskit gridsynth_rz/gridsynth_unitary、Quantinuum/grid_synthesis（Rust，实时 profiling）；PennyLane Catalyst 也集成 gridsynth pass。使用场景：FT 编译器把 Rz 分解为 Clifford+T（表面码/魔态注入前的最后逻辑阶段）、T-count 优化研究。
- 与本文关系：GridSynth 是组合对象（评估 PhasePoly 在 FT 管线中的最佳位置），非被修改工具。
- 补充（TACO 论文）：TACO 把 GridSynth 作为默认合成后端（替代 Qiskit 默认 Solovay-Kitaev），默认误差容限 ε=10⁻¹⁰（对比 O3LS 用 10⁻⁵）；图 3c 显示对随机 1-qubit unitary，GridSynth 用 332 个 T 门达到 <10⁻¹⁰ 误差，而 Solovay-Kitaev 需 5 万+ 门。TACO 的价值在于 GridSynth 只优化单比特 Z 轴旋转（Rz）且对"需合成的 unitary 数"敏感，因此 TACO 先做 FTQC 导向动态分解（降低中间电路 Rz 门数）再合成，比直接合成减少 1.26× 需合成 unitary、490× T 门（Table VI：QFT T 门 2,623,881→9,529，QPE 275,356→411）；18 比特 QFT 全转译 <1s 而 GridSynth 单独合成需 16.1s。实现：NWQEC 内嵌高性能 C++ gridsynth 后端（约 20× 加速），Qiskit 亦集成（ross_selinger.py / rsgridsynth Rust 参考实现 qiskit-community/rsgridsynth）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
