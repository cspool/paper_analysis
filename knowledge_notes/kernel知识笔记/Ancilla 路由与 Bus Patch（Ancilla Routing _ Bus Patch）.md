## Ancilla 路由与 Bus Patch（Ancilla Routing / Bus Patch）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Ancilla 路由是晶格手术中为多 patch 测量提供"总线"的空间机制：ancilla patch（路由空间，即 bus）是一组被保留用于临时初始化和测量的空 tile，数据 patch 的 X/Z 边缘须邻接路由空间才能被测量。Bus patch 列表 L_{P_i} 是执行 Pauli 算子 P_i 所需的最短路由补丁集合。路由长度直接决定：①PPM 错误率（路由空间越大错误越多，P_PPM 由路由空间与码距决定）；②idle 记忆错误（ancilla 路径越长，patch 闲置时间越长）；③空间开销（tile 数）。因此最小化路由路径是布局设计（squeezed 布局）与调度（Dijkstra 最小 bus）的共同目标。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 O3LS 松散调度 Algorithm 2 Step 5：对可执行 P_i 用 Dijkstra 依次求所需 patch 间最短路径，之前已识别的路径节点视为零成本（复用），得到最小 bus patch 列表 L_{P_i}——最小化总路由路径长度既增加其他操作的并行机会又降低 LER，每算子复杂度 O(|B|²)。布局设计侧：评分函数 S(B)=C(B)×(N_x(B)+N_z(B)−α_e·N_e(B)) 中，C(B)=路由连通存在性（ancilla 的 X/Z 边缘都须连到路由空间以支持 Y 测量）、N_e=数据 patch 到路由空间的边缘数（α_e∈[0.1,0.3] 密度因子）。例子：Z_0Z_1Z_2Z_3Z_4 测量在 standard 布局需 6 个 ancilla 路由 patch，O3LS squeezed 布局只需 5 个；相对 SPARO 布局平均 ancilla 路由空间降 17.35%，完整 O3LS 栈再降 27.17%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译时布局与调度决策：布局搜索决定每个 patch 的位置与边缘暴露（squeezed 布局缩短路由），调度时 Dijkstra 决定每步 PPM 的 ancilla 路径；物理上 ancilla patch 就是表面上被临时激活做稳定子测量的空 tile。评估指标：ancilla patch 长度/路由空间（每时间片记录）、时间步数、LER。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
