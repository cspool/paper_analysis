## Qubit Mapping / Routing（量子比特映射/路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Qubit mapping/routing 是量子编译器（如 Qiskit、TKET）中把逻辑线路适配到真实硬件拓扑约束的编译阶段：先做 qubit mapping（把逻辑量子比特 q_i 赋给物理量子比特 Q_i 的初始布局 initial layout），再对不满足最近邻连接约束的两比特门做 qubit routing——通过插入 SWAP 门动态交换两个物理量子比特上的状态子空间，把非相邻的逻辑量子比特"搬"到一起执行（SWAP-based routing，对应经典编译器中的寄存器分配与指令调度）。由于超导平台上两比特门只能作用于最近邻物理 qubit 对（Google 2D 方阵、IBM 2D heavy-hex），路由是 NISQ 与低层容错编译的必经阶段，其引入的 routing overhead 通常使门数与深度增加 2-5×（相对 SOTA 可扩展方法）。问题形式化为：输入逻辑线路 DAG + 硬件耦合图，输出每门都满足耦合图边的物理线路（见 CANOPUS 论文 Fig.2）。
- 关键方法谱系：Zulehner 等的 A* 分层方法（最小化并发 CX 层的 SWAP 开销）、Li 等的 SABRE（双向迭代启发式，工业标准，Qiskit 默认）、TOQM（A* 深度驱动）、Ddroute（深度驱动 DAC'25）、Lightsabre（轻量增强版）、BQSKIT（数值优化 rebase 范式）、CANOPUS（本文，ISA-aware 规范表示路由）。衡量指标：路由后门数/深度相对路由前的比值（routing overhead）。
- 传统模型缺陷（本文动机）：把 SWAP 固定按 3 个 CX 计费（SWAP = CX·CX·CX），完全忽略后端 native 门（√iSWAP、iSWAP、ZZ(θ) 族）的实际合成能力与物理时长，导致路由决策与真实执行成本脱节、优化空间被系统性高估。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架内运转流程（SABRE 型，CANOPUS 继承其骨架）：输入逻辑线路 DAG + 耦合图 → ① 分层：提取 front layer F（当前无未解决依赖的可执行门集合）与 lookahead extended set E（后续若干层扩展集）；② 逐层处理：剥离可执行门（两 qubit 已在相邻物理位），对非合规门枚举候选 SWAP（作用于耦合图每条边）；③ 启发式评分：对每个候选 SWAP 计算代价（如 SABRE 的 H = Avg{dist[i,j]}_F + k_E·Avg{dist[i,j]}_E，即 F 与 E 中逻辑交互对应物理 qubit 的平均最短路径距离），选最小者插入并更新逻辑-物理映射 π；④ 双向迭代：正向 pass 后反向重跑校准初始映射，迭代多轮取最优。
- CANOPUS 的增强流程：在评分时把"固定 3×CX 的 SWAP 成本"替换为 ISA-aware 的边际合成成本 c_g = COST(SWAP·U) − COST(U)（U 为 last mapped layer L 中同物理 qubit 对的前驱 2Q 门，SWAP mirroring），加上基于 wire duration record D 的深度增量 Δdepth 与差分拓扑距离项，统一启发式 H = w_g·c_g + w_d·Δdepth + (ΔAvg{dist}_E + k_E·ΔAvg{dist}_E)·c_swap，从而在路由阶段同时优化门数与深度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Qiskit transpiler 中 `transpile(circ, coupling_map=..., routing_method='sabre', optimization_level=...)`（routing_method 可选 sabre/basic/stochastic/lookahead/none）；SABRE 为默认。TKET 的 routing（qiskit 生态外）也有类似 pass。CANOPUS 开源实现为 Qiskit TransformationPass（CanopusMapping），`pip install canopus-quantum` 后接入既有 transpilation pipeline；`python route_demo.py`/`route_qft.py <n>` 可对比 SABRE 与 CANOPUS。
- 在 CANOPUS 实验中的例子：216 个 case（6 ISA × 3 拓扑 × 12 benchmark）上 CANOPUS 对所有 ISA-topology 组合取得最低 routing overhead（相对 SABRE 平均降 16.06% Ccount/26.44% Cdepth）；1D chain 上 QFT 路由达到理论最优（qft_6：#Can 15、Depth2Q 9）。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
