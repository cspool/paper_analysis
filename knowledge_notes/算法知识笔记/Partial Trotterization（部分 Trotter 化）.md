## Partial Trotterization（部分 Trotter 化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Partial Trotterization 是本论文提出的新哈密顿量编译范式：不再对每个哈密顿量项单独指数化（全 Trotterization），而是把非对易项分组进 partition（本论文选每组至多 n=3 个不同 qubit，即 8×8 unitary），把"组内多项之和"的指数 e^{itΣH_i} 作为一个整体直接编译。例：全 Trotter 把 e^{i(H_1+H_2+H_3)t} 展开为 e^{iH_1t}e^{iH_2t}e^{iH_3t}，partial Trotterization 则可展开为 e^{i(H_1+H_2)t}e^{iH_3t}。核心收益：BCH 公式下误差主导项由组内 commutator 构成，分组后这些组内 commutator 随组的指数化一起消失——误差从全 Trotter 的 sum_{i<j}|[H_i,H_j]|Δt²/2 降为仅跨组项 sum_{A≤B}|[H_A,H_B]|Δt²/2（Eq.7-9），达到相同精度所需 Trotter 步数大幅下降。
- 理论结果：误差缩减随 group 大小 ~n_A² 组合增长（组内 commutator 数以 n_A² 量级被消除）；一阶/高阶 Trotter 下电路深度随 group 大小呈二次缩减（quadratic reduction）。实测（图 8）：一阶 Trotter 10 步下误差随每 unitary 的 qubit 数（1→3）急剧下降；非对易对占比不随 lattice 增大（图 9，n=3 与 n=5），保证常数 partition 大小的可扩展性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pipeline 第一步（Algorithm 1 Greedy Partitioning，Table 1 示例）：
```
输入: [X_3, X_1X_2, Z_3Z_4, Z_1]  # 4-qubit 自旋哈密顿量项
排序: 按最高 qubit 索引，平局按权重 -> [Z_1, X_1X_2, X_3, Z_3Z_4]
贪心分组: 每组并集 qubit 数 <= 3
  -> partition1 = {Z_1, X_1X_2}, partition2 = {X_3, Z_3Z_4}
输出: U_1 = e^{i(t/N)(Z_1+X_1X_2)}, U_2 = e^{i(t/N)(X_3+Z_3Z_4)}  # 8x8 unitary
```
- 误差对比例子（Eq.3 vs 4）：H=H_i+H_j+H_k+H_l 四互不对易项，全 Trotter 误差 ∝ 6 个 commutator；partial Trotterization 分组 {(H_i,H_j),(H_k,H_l)} 后误差只余 4 个跨组 commutator（2 个组内 commutator 被消除）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现需要解决两个挑战（论文 Section 3）：(1) 如何有效划分项（高密度 partition）——用"高层算子表示"做划分（high-level circuit partitioning），排序+贪心，比门级电路划分更稠密；(2) 如何高效编译分组指数 e^{itΣH_i}——用 MCTS unitary 分解（见编译框架库 MCTS 条目），因为逐项实现会退回 vanilla Trotterization 失去误差收益，通用 unitary 分解（QSD 等）门数又太高。使用时：输入 H、时间 t、步数 N、partition 大小 n（论文取 n=3），输出优化线路；n 越大误差收益越大（组内 commutator 以 ~n_A² 消除）但 unitary 分解难度上升。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
