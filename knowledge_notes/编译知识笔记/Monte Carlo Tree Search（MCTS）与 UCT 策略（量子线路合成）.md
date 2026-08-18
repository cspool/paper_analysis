## Monte Carlo Tree Search（MCTS）与 UCT 策略（量子线路合成）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Monte Carlo Tree Search（MCTS）是解决序贯决策问题（环境信息少、搜索空间大）的强化学习/启发式搜索算法，四阶段循环：Selection（按策略从根遍历到未探索节点）→ Expansion（从未探索节点选动作扩展出新状态）→ Simulation（从新状态随机快速推演到终态并评估）→ Backpropagation（把评估值沿路径回传更新节点值）。策略平衡 exploration（探索新解）与 exploitation（利用已知优解）；经典策略是 UCT（Upper Confidence Bound applied to Trees）Q 值公式。本论文把 MCTS 用于量子线路合成：把部分 Trotter 化的 8×8 unitary（n=3 partition 的指数）分解为 CNOT+单比特门线路，替代通用 unitary 分解（QSD 等门数过高）与纯数值合成（搜索空间过大）。
- 关键设计：树节点状态只表示"纯 CNOT 线路"（不表示单比特门角度），动作=可追加的 CNOT 门。这压缩了分支因子 b 与深度 D（角度是连续空间，若进入状态会使搜索不可行），提高 coverage≈N_visited/b^{D+1}（Eq.4）。单比特门角度在 Simulation 阶段用可微方法（Gauss-Newton）求解（见 CNOT 骨架条目）。值函数（Eq.6）：误差 <1e-8 时值=−#CNOT，否则值=−误差（近似误差）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 Kernpiler 编译流水线中的位置：Hamiltonian partitioning（Algorithm 1）→ conflict graph 分组/重排/合并（Algorithm 2）→ MCTS 重写（4.4 节）→ Qiskit transpiler level 3 后优化。MCTS 重写流程（图 4）：
```
目标: 8x8 unitary U = e^{i(t/N)(H_a+H_b)}   (partition 指数)
1) Selection: UCT 策略从根（空 CNOT 线路）选未探索节点
2) Expansion: 从门集选一个 CNOT 追加 -> 新节点（纯 CNOT 线路）
3) Simulation: 随机追加 CNOT 到固定长度；在 CNOT 间随机插入固定数量单比特门
   用 Gauss-Newton 最小化 || Π x_i(θ_i) - U ||_2 求单比特门参数 θ
   值 = -#CNOT (误差<1e-8) 或 -误差
4) Backpropagation: 沿路径更新 Q 值
重复直到找到最优线路 -> 输出
```
- 实际效果（图 5）：MCTS 重写相比 Qiskit synthesis/BQSkit 各 benchmark 次优方案 U3 减 60%(峰值)/33%(平均)、CNOT 减 67%/34%、depth 减 60%/32%；相比 BQSkit 更快，因为用高层哈密顿量结构信息（可设目标如"找 10 门解"）缩小搜索空间。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MCTS 在 NVIDIA A100 GPU（80GB）上实现（PyTorch 2.5.1 + CUDA 12.1），PyTorch 提供矩阵/微积分运算（Gauss-Newton 需张量操作与 L2 范数梯度）；UCT 是标准树策略。使用时按论文四阶段；对每个 partition 的 unitary 独立搜索。局限：MCTS 只优化 CNOT 数（cost 函数不含 U3），单比特门靠过参数化+transpiler 后优化，导致 U3 门数相对偏高（Ising 模型明显，图 6 讨论）。相关外部工作：MCTS 也被用于量子线路变换/路由（Zhou-Feng-Li 2022）与 Hamiltonian simulation 合成（MonteQ，2026）。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
