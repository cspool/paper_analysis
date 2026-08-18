## CNOT 骨架搜索与单比特门过参数化（CNOT Skeleton + Overparameterization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是 Kernpiler MCTS 合成器的核心设计：把线路合成拆成"CNOT 骨架"（entanglement structure，即 CNOT 的连接/排列，决定线路的纠缠结构）与"单比特门参数"（连续角度）两个子问题。理由：(1) 纠缠结构是合成中最难的部分，而单比特门连续参数化有平滑的优化 landscape，可用可微方法求解；(2) 若把角度也放进 MCTS 状态，动作空间连续（branching factor 发散）且深度大，coverage≈N_visited/b^{D+1} 无法覆盖搜索空间。因此树节点状态只含 CNOT，Simulation 阶段在 CNOT 间随机插入固定数量单比特门（过参数化 overparameterization）后用 Gauss-Newton 最小化 L2 范数求角度。
- 过参数化的作用：比最小参数化线路有更平滑的 cost landscape（减少局部极小），更容易收敛；优化后用 Qiskit transpiler level 3 把单比特门串折叠为单个门（(u3,cx) 门集最优线路）。局限（论文自述）：cost 函数只含 #CNOT（Eq.6），未优化 U3 数，导致 U3 门数偏高（Ising 模型最明显，图 6）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 MCTS 合成中的流程（4.4 节，图 4）：
```
Selection: 选未探索节点（= 部分 CNOT 线路）
Expansion: 追加 1 个 CNOT -> 新节点
Simulation:
  随机追加 CNOT 至固定线路长度
  在 CNOT 间随机插入固定数量单比特门（过参数化）
  Gauss-Newton: θ* = argmin_θ || Π_i x_i(θ_i) - U ||_2   (Eq.5)
  值 R(x) = -#cnot 若 E(x)<1e-8，否则 -E(x)                (Eq.6)
Backpropagation: 沿路径更新 Q 值
输出: 最优 CNOT 骨架 + 参数化线路 -> Qiskit transpiler L3 折叠
```
- 效果：相比纯 CNOT 树/链分解（如 Paulihedral/Rustiq 用同时对角化），本方案无需反复基底变换，但对 Ising 等低 CNOT 模型会在"odd/sandwiched"位置产生更多 U3（缺少 CNOT tree 带来的 U3 削减）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Gauss-Newton 在 PyTorch 2.5.1 + CUDA 12.1（A100）上做张量优化（L2 范数梯度迭代）；Qiskit transpiler level 3（u3/cx basis、all-to-all）做最终门折叠。使用时：每个 8×8 partition unitary 一次合成；搜索策略用 UCT 平衡探索/利用，误差阈值 ε=1e-8 决定值函数在"门数优先"与"误差优先"间切换（低于阈值后精度收益递减、门数成为主导目标）。结论中论文指出：MCTS 反复发现的 CNOT 骨架/纠缠 motif 可启发非 RL 的启发式或图基合成算法，是后续研究方向。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
