## Clifford+T 门集与 magic-state distillation / cultivation（容错通用门集）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Clifford+T 是容错量子计算（FTQC）的标准通用门集：Clifford 群门（H、S、CNOT，由 {CNOT,H,S} 生成）+ 非 Clifford 的 T 门（π/4 Z 旋转）。Clifford 门可由稳定子码透明地容错执行，T 门不能直接容错，需 magic-state 辅助：预制备 |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2 类非 Clifford 态（经 distillation/cultivation 提升保真度），再经 teleportation 注入实现 T。
- 传统 FT 成本模型：T 门昂贵（magic-state distillation 需要大量物理资源与空间时间体积），因此 FT 编译长期聚焦最小化 T 门数/深度（T-count/T-depth 优化，如 T-par 的 matroid partitioning、phase polynomial 的 T 优化）。论文引用的新进展：magic-state cultivation（2024-2025，低开销直接培育高保真 magic state）与更新资源模型表明 T 与 CNOT 的成本日益可比——CNOT 不再是"免费"门，需与 T 联合优化。
- 与相位多项式关系：Clifford+T 门集中，{CNOT,Rz} 区域自然涌现——CNOT 算 parity，T/S/Z（Rz 的子集）累积相位；Fig.1 显示 FT 基准中 CNOT 与 T 数量相当，二者共同主导。因此"联合减 CNOT 与 Rz/T"（PhasePoly 的目标）直接降低 FT 资源成本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FT 编译 pipeline 计算过程：逻辑线路 → 逻辑优化（PhasePoly 减 CNOT/Rz）→ 任意 Rz 分解到 Clifford+T（GridSynth/Ross-Selinger：Rz(θ) → H、S、T 序列，T-count 在 O(log(1/ε)) 内近最优）→ 容错编码（surface code）→ 资源估计（Azure Resource Estimator：按 magic state 制备成本、CNOT 空间时间成本、测量/路由开销估算 wall-clock 与物理 qubit）。论文 Q5 流程：(A) GridSynth→PhasePoly vs (B) PhasePoly→GridSynth 两种顺序；结论 B 更优——先 PhasePoly 化简大 {CNOT,Rz} 区域，再 GridSynth 引入额外 H 门（会把 phase polynomial block 再切碎、限制后续 rotation merging 机会）。
- 资源模型例子：surface-code nearest-neighbour 架构下，PhasePoly 减 CNOT 44.62% FT wall-clock（vs Quartz 11.99%、QUESO 31.80%，Fig.17）；T-count 变化温和（主要影响 HWPA 等结构化电路，深度 ~10% 减）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Rz→Clifford+T 用 GridSynth（Ross & Selinger arXiv:1403.2975；实现：Qiskit gridsynth_rz、Quantinuum/grid_synthesis Rust 版、Haskell gridsynth）；FT 资源估计用 Azure Quantum Resource Estimator（微软云服务，surface-code 假设）；T 优化工具 T-par、PyZX 等。magic-state cultivation 见 2024-2025 文献（如 arXiv:2409.17543 等，论文引 [25-27]）。
- 与本文关系：PhasePoly 在 Clifford+T 合成前运行收益最大，且其联合减 CNOT/Rz 在"CNOT 成本可比 T"的新资源模型下价值上升。
- 补充（O3LS 论文）：O3LS 在 PBC 框架下使用 Clifford+T 门集——任意旋转经 GridSynth（qiskit-gridsynth-plugin，基于 Ross-Selinger arXiv:1403.2975，合成误差容限 10⁻⁵）分解到 Clifford+T；随后转译为 Pauli product rotations（S=Z_{π/4}、T=Z_{π/8}、H=Z_{π/4}X_{π/4}Z_{π/4}、CNOT=(Z⊗X)_{π/4}(I⊗X)_{−π/4}(Z⊗I)_{−π/4}），Clifford 门按 Pauli 映射规则吸收进最终测量。T 门成本（magic state 消费）仍是开销来源，但 O3LS 关注点从"最小化 T-count"转向"布局/调度/合成联合优化时间步与空间"；Y-synthesis 正是针对 PBC 转译后 Y 算子分解的合成优化（对应本条目"相位多项式"外的另一条合成路径）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
