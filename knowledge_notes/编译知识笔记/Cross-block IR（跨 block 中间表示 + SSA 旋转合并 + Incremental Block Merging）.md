## Cross-block IR（跨 block 中间表示 + SSA 旋转合并 + Incremental Block Merging）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 通用量子线路中 H 等换基门是 block barrier，把线路切成多个 phase polynomial block；先前方法（Gray-Synth、单 block 贪心）只能做 block 内优化，错过跨 block 的奇偶性复用与旋转合并。PhasePoly 的 Cross-block IR 把相邻 block 合并为更大 phase polynomial region：(1) SSA 风格 qubit-state 重命名——每个输入 qubit 态与每个 H 后状态分配 fresh SSA ID，每个 Rz 打上所在 qubit 态的 SSA ID，同 SSA ID 的 Rz 全局合并（"整线路 rotation merging"，正确性判据简化为同 ID 即合并；先前的 anchors/terminal points 方法复杂且保证弱）；(2) 跨 block parity 矩阵——post-H qubit 状态作为新行以 inactive 状态存在于 IR，直至其 pre-H producer 行被消除并插入 H 才激活；(3) 线性相关性（rank）检验——消除 pre-H 行需满足：该行 phase 全零（无待发射旋转）+ 列隔离（该列为单位向量）+ 行隔离（该行为单位向量）；条件 (2) 由 (1)+(3) 推出，等价于目标单位向量 v ∈ span(候选行)，用 rank(M∪{v})=rank(M) 检验（否则剪枝），保证可合成性；(4) Incremental Block Merging——沿 block DAG 前向遍历，先单 block 优化作稳定基线，渐进扩大合并组（Group k 至 7），只保留改善的步骤，避免过合并回归（ham15_med：Group 7 退化 694 门而 Incremental 达 656 门/325 CX）。效果：MCX 中约一半 Toffoli 结构从 ~4 CNOT 降到 3；9/28 电路受益于 cross-block。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Fig.9 两 block 例子）：Block1（{q₀,q₁,q₂}）与 Block2（{q₀,q₁,q₃}）共享 {q₀,q₁} 故可合并 → ① SSA 命名：输入 q₀,q₁,q₂ 各一 ID；H 前 q₂ 上的 parity q₁⊕q₂ 传输到 q₂ 线上；H 后创建新 SSA 状态 q₃（output-parity 变 {q₀⊕q₃}）；② 合并为单 block：q₃ 行 inactive 锁住，A* 中不作为 row pair 候选；③ 消除 q₂（pre-H 行）：先验该行 phase 全零、用 rank 检验 v（q₂ 目标行）∈ span → 解 Mα=v（GF(2)）得 witness set S；若待消行 t∉S，先 CNOT(j,t) 把 t 纳入 S'（不改变 span），再对 i∈S'∖{t} 做 CNOT(t,i) 实现行隔离（row_t ← ⊕S' = v），对 k≠t 且第 t 列有 1 的做 CNOT(k,t) 清列（Fig.10 四步矩阵演化）；④ 删行 t、插入 H、激活 post-H 行 q₃。结果：5 CNOT+3 T → 4 CNOT+1 T（跨 block 的 T 合并 + CNOT 复用）。
- 作用：把"block 内局部最优"提升为"跨 barrier 全局最优"，是 PhasePoly 相对 Gray-Synth/单 block 方法在大电路上持续领先（Fig.5 MCX 19→499 qubit 线性增长 vs 重写框架饱和）的结构性原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：在 PhasePoly 内与耦合矩阵 A* 集成（仓库 https://github.com/ruadapt/PhasePoly）；参数 G 控制合并组大小，Incremental 策略自动选择。block DAG 按 qubit 依赖建边，仅共享多比特交互的相邻 block 才评估合并（控制复杂度）。正确性由 rank 检验 + 最终端到端等价验证保证（Qiskit unitary 对比 <8 qubit + MQT QCEC 全电路，全部通过；mod_adder_1024 因硬件限制除外）。
- 与相关方法对比：Amy & Lunderville（arXiv 2024）用关系程序分析在控制流/Toffoli 上发现额外旋转合并机会，与本文互补（本文还暴露两比特门间的 parity 关系）。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
