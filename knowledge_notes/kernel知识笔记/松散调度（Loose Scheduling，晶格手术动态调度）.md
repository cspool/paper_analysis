## 松散调度（Loose Scheduling，晶格手术动态调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 松散调度是 O3LS（ISCA 2026）提出的晶格手术执行调度策略："loose"指调度的灵活性——在表面码 patch board 上，不按固定的预定义模式执行，而是根据电路上下文与布局约束，动态重指派 patch 功能（位置/朝向）、消除冗余 patch 移动与不必要操作，最小化总时间步。它针对的痛点是静态调度（如 SPC/LAPBC）：固定策略对所有电路统一施加"多 patch 测量前必做 patch 旋转对齐 X/Z 算子"，忽略了可免旋转的场合；在紧凑/不规则布局中 patch 旋转（3 时间步）可占 >50% 时间步（O3LS Fig.7），是空间受限布局的主要瓶颈。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Algorithm 2（Loose Scheduling Algorithm）伪代码：
  ```
  输入：Pauli 算子序列 S={P_1..P_l}，qubit 数 n，board B
  输出：可执行操作序列 S'
  1 初始化 S'={}
  2 从 S 构建 Pauli DAG G
  3 while G 非空 do
  4    for P_i in G.frontier 且 P_i 可执行 do
  5        对 P_i 用 Dijkstra 求最短 bus patch 列表 L_{P_i}（已占路径视为零成本节点）
  6        在 L_{P_i} 上执行 P_i，S' ← S' + P_i
  7    end for
  8    从 G.frontier 弹出一个 Pauli 算子 P_i
  9    while P_i 不可执行 do
  10       枚举 B 上全部候选 patch 操作 O_B
  11       选 r(o_b, P_i) 最大的 o_b ∈ O_B
  12       在 B 上执行 o_b，S' ← S' + o_b
  13   end while
  14 end while
  15 return S'
  ```
  奖励函数 r(o_b,P_i) 三部分：①施加 o_b 后 board 状态 B_o 中使 P_i 可执行（存在有效路由路径）的数据 patch 数；②破坏数据 patch 连通性的操作奖励为 0（连通性为后续 lattice surgery 必需）；③同奖励时优先低时间开销的操作。每步至少增加一个满足执行要求的 patch，总复杂度 O(n²)；每个可执行算子的 Dijkstra bus 路径复杂度 O(|B|²)。例子（Fig.4）：执行 Z_0Z_1Z_2Z_3Z_4 测量，松散调度只把 q_0 下移并旋转暴露不同边缘（而非整 patch 旋转），compact 布局下 6 个 ancilla patch → squeezed 布局 5 个，时间步显著减少。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器调度 pass，运行在 O3LS 编译流水线末端（布局搜索 → Y-synthesis → 边感知初始映射 → 松散调度），输入是 PDAG frontier + board，输出是时间步序列与每时间片 patch 布局。效果：相对 SPC 时间步降 36.07%（compact）/24.76%（standard）；相对 LAPBC 平均降 35.10%（最大 80.6%）；与 SPARO 调度组合再降时间步 78.24%、路由空间 27.17%、LER 77.1%。可与高并行策略（LAPBC 风格）集成再降 9.31% 时间步。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
