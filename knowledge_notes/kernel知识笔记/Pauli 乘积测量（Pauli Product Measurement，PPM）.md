## Pauli 乘积测量（Pauli Product Measurement，PPM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PPM 是晶格手术中的核心原子操作：在表面码 patch board 上测量多 qubit Pauli 算子（Pauli 串乘积，如 Z_0Z_1Z_2Z_3Z_4）的期望值/本征值。实现机制：当目标 patch 的相关边缘与 ancilla 路径（路由空间）相邻时，初始化 ancilla patch A、合并 A 与目标 patch 群（沿共享边界做稳定子测量）、测量、再分离，时间成本 1 时间步（round of code cycles）。多 patch π/4 与 π/8 测量用标准 gate teleportation 协议（Litinski [34]）。PPM 是 Pauli-Based Computation（PBC）编译模型的执行原语——Clifford+T 线路转译后以 PPR/PPM 序列执行。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- PPM 执行流程（在 O3LS 松散调度中，Z_0Z_1Z_2Z_3Z_4 为例）：①调度器确认该 PPM 可执行（各目标 patch 边缘可经路由空间连通）；②Dijkstra 求最短 bus patch 列表 L（已占路径节点视为零成本），得到最小 ancilla 路由补丁集；③在 L 上初始化 ancilla（0 时间步）→ 依次合并目标 patch（1 时间步）→ 测量 → 分离。伪代码层面即 Algorithm 2 的 Step 5-6。PPM 的错误率主要由路由空间与码距决定；在 LER 分层模型中 PPM 是 p_layer 的组成部分之一（P_PPM）。squeezed 布局下 Y 型 PPM 需先经 Y-synthesis 分解为 X/Z 组合（因 X/Z 不能同时访问）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译产物中的可执行指令（lattice surgery instruction，含目标 patch 集与 ancilla 路径），物理上由表面码稳定子测量电路执行。评估：STIM 仿真（d=9、p=10⁻³ 电路级去极化噪声，Monte Carlo ≥10⁶ 次）表征每个原子操作错误率，PyMatching 2 解码。O3LS 中 PPM 是调度与布局设计的核心目标——布局评分函数保证"所有数据 patch 可测量"（连通性 C(B)），即任何 PPM 都可执行。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
