## Pauli DAG / O3LS-IR（Pauli 有向无环图中间表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- O3LS-IR 是 O3LS 引入的中间表示：Pauli 算子表示为节点，构成 Pauli 有向无环图（PDAG），记录依赖关系以确定执行顺序并暴露并行性。每个节点含：①旋转角（也可标识该节点是否为测量）；②Pauli word（Pauli 算子）；③前驱节点；④后继节点。PDAG 构造规则：两节点 P_i、P_j 之间存在有向边当且仅当 (1) 存在至少一个 qubit q 使 P_i 与 P_j 都不是恒等 I；(2) P_i 与 P_j 之间没有其他 Pauli 算子 P_k 在 q 上非平凡作用。入度为 0 的节点对应可执行 Pauli 算子；执行后移除节点，重复直到全部处理完，保证依赖解析正确。作用：支撑 Y-synthesis（吸收抵消判定）、边感知映射（旋转需求估计）与松散调度（frontier 遍历）三模块共享同一 IR。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 构建与使用流程（编译框架视角）：输入转译后的 Pauli 算子序列 S → 逐算子扫描建立节点与依赖边（规则如上）→ 得到 DAG。调度循环（Algorithm 2 Step 3-8）：while DAG 非空：对 frontier（入度 0 集合）中可执行算子调度执行；弹出 frontier 中一个算子，若不可执行则循环选 patch 操作（奖励函数）直到可执行。例子：序列 Z_0Z_1 → Z_0Z_2（共享 qubit 0）间有依赖边（qubit 0 非平凡），而 Z_0Z_1 与 X_2X_3（不共享非平凡 qubit）无依赖可并行。Y-synthesis 用 PDAG 判定分组吸收：P_i 的前驱已分解，检查分组算子能否被前驱吸收；对后继则检查是否存在潜在 Y 分解机会。复杂度：PDAG 构建与 Y-synthesis 每算子 O(n)（检查 n 前驱 + n 后继），总体 O(nl)。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器内部 IR 数据结构（DAG，节点含旋转角/Pauli word/前后继指针），在转译后立即构建，供合成/映射/调度三模块复用。编译时间：O3LS 借助 O3LS-IR 显式 Pauli 变换比 SPC 更快、与 LAPBC 相当，且随 qubit 数多项式扩展（Fig.22）。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
