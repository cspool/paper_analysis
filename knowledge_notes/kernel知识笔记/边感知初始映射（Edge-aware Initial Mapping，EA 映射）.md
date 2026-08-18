## 边感知初始映射（Edge-aware Initial Mapping，EA 映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 边感知初始映射是 O3LS Module 4：在把逻辑 qubit 映射到数据 patch 时，用 Pauli DAG（PDAG）估计每个 qubit 的旋转需求（预期旋转频率），把旋转需求高的 qubit 优先映射到同时邻接 ancilla patch 的 X 与 Z 边缘（双边缘）的 patch。动机：patch rotation 是 3 时间步的昂贵操作，在 squeezed 布局中频繁发生；若高频切换 X/Z 算子的 qubit 落在只暴露单边缘的 patch，会反复触发旋转。映射到双边缘 patch 后多数旋转可省。复杂度 O(n log n)（PDAG 中旋转需求提取 O(n) + 两次 quicksort）。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程：①构造 PDAG 时对每个 qubit 统计其参与的 Pauli 算子中 X/Z 类型的切换次数，得到旋转需求排序；②统计 board 上每个 patch 暴露的边缘类型（X-only / Z-only / 双边缘）；③两次 quicksort：qubit 按旋转需求降序、patch 按"边缘丰富度"降序，一一对应放置。例子：加法器电路某 qubit 频繁参与 Z_0Z_1 与 X_2X_3 测量 → 映射到 ancilla 旁的双边缘 patch；而只参与 Z 测量的 qubit 可放 Z-only 边缘 patch。适用性：高度紧凑布局中位置变化难以暴露新边缘（收益依赖电路结构）；稀疏布局中每 patch 天然双边缘（收益小）；介于两者之间的 squeezed 布局收益最大。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译流水线中的初始映射 pass（在松散调度之前、Y-synthesis 之后运行）。效果：相对 SPARO [28] 的 greedy mapping，时间步减少 15.0%、LER 减少 8.4%；在完整 O3LS 栈（O3LS-2+3+4）中贡献平均 38.62% 时间步、35.17% LER 的改善（ablation Fig.21）。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
