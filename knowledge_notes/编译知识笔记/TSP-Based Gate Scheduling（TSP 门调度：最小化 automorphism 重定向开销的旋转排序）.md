## TSP-Based Gate Scheduling（TSP 门调度：最小化 automorphism 重定向开销的旋转排序）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 把可对易门（旋转）的执行顺序选择建模为旅行商问题（TSP）的编译优化。前提：triorthogonal 蒸馏协议的所有 exp(iπ/8·P) 旋转两两对易，任意重排不改变逻辑信道；而相邻两次原生测量之间的重定向开销（把 LPU 从测 P 的配置变换到测 Q 需施加的 automorphism 序列）随顺序剧烈变化。建模：每个不同 Pauli 标签 P 为节点 v；有向完全图 G=(V,E)，边权 w(u,v)=从 u 的测量配置变换到 v 的 automorphism 开销（轮数/时延/任何硬件度量）；任何顺序 σ 总开销 $C(\sigma)=\sum_i w(v_{\sigma(i)},v_{\sigma(i+1)})$。最小化 C(σ) = 求固定端点的最小代价 Hamiltonian path——TSP 变体。量子编译即 TSP 的观点可追溯 Maslov 的 NP-complete 证明与 Paler-Zulehner-Wille（arXiv:1806.07241，QST 2021）把 qubit allocation+gate scheduling 建模为环面上 TSP 的工作。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译流程（本论文 Sec. IV-C）：
  ```
  输入: 协议旋转集合 {P_1..P_15} + automorphism 代价矩阵 W（每码实例预计算一次）
  TSP 求解: 最近邻初始化 + 2-opt/3-opt 细化，或 warm-start MILP
          （15-to-1 仅 15 节点，秒级求得近最优/最优）
  输出: 旋转执行序 σ*，使相邻测量共享相似 automorphism 动作、
        聚类"同向"旋转、减少每次重定向的 automorphism 轮数
  代价: <3 s（Table II）；矩阵跨工厂周期复用 → 边际调度开销可忽略
  ```
  效果：Fig 9(c) 相对 [46] 原门序减少 automorphism 轮数；每省一轮 automorphism 即省 14 个 timestep 与相应逻辑错误机会（Table III/IV 的全部 τ_i 已含该优化）。注意 TSP 只改顺序不改门与角度，正确性无损。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 论文代码：prj_msd/30min_aut/min_aut.py。通用迁移：任何"原生算子重定向有代价 + 算子可重排"的编译问题（如 NISQ 芯片上 remote CNOT 的路由序、qubit allocation）都可化归 TSP/哈密顿路径变体求解；小实例用 2-opt/3-opt/MILP、大实例用元启发式。与 [31] 的 Clifford 共轭编译组合使用时，TSP 适用于 direct injection 这类相邻旋转之间无需 pivot 中间测量的方案。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
