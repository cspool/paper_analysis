## Randomized Compilation（随机化编译）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Randomized compilation（随机化编译）是把量子线路中的相干误差（coherent error，系统性、可累积）转化为随机/随机化误差（stochastic error，随运行平均抵消）的技术，源于 randomized compiling 与 Pauli twirling 思想（Wallman & Emerson 2016；早期理论框架 [4,48,46,47,17,18]）。在 product formula 中，随机化编译通过"每步 shuffle Trotter 项的顺序"实现：让快速变化的演化项平均掉错误项，得到更好的误差缩放（Campbell 2019 randomized compiler；Childs et al. 2019 randomization）。本论文在 Trotter step 内的 group 间随机 shuffle 项顺序（图 3 Step 3），把"每个 Trotter step 都重复相同的近似误差"（coherent noise）变为"每个 step 不同的随机误差"（stochastic noise），从而在平均意义下降低误差。
- 关键区别：shuffle 只在 group 内部进行（不跨 group），保证 group 结构（跨 step 合并、边缘 group 位置）不被破坏——这是本论文在随机化与编译优化之间保持兼容的设计选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文用法（图 3 Step 3 + Algorithm 2）：
```
对每个 Trotter step k:
  groups_k = GreedyCommutingGroups(conflict_graph(H_k))  # 互交换 group
  对每个 group: randomize(顺序)                            # 组内随机 shuffle
  # 保持 group 边界: 不 shuffle group 之间的顺序
相邻 step 重排 + 跨 step 合并对易项 -> 输出 modified Trotter steps
```
- 效果：每个 step 的乘积顺序不同 → 近似误差的高阶项随机化，避免相干累积；与"把两个最大 group 放 step 两侧、相邻 step 翻转"（使相同 group 相邻可合并）协同工作，zero 额外近似误差地降低 unitary 数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：通用 randomized compiling 在运行时对每层应用随机 Pauli 伴算（gate twirling），在编译期/运行期都可做；本论文把它限制在"Trotter step 内 group 内顺序 shuffle"这一更轻量的形式，作为 Algorithm 2（REORDERTROTTERSTEPS）的一步。使用时与分组/重排/合并流水线集成，无需额外硬件或测量开销。

涉及论文标题：
- Kernpiler: Compiler Optimization for Quantum Hamiltonian Simulation with Partial Trotterization
