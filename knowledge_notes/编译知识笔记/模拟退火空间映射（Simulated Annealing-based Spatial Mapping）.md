## 模拟退火空间映射（Simulated Annealing-based Spatial Mapping）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 模拟退火（SA）是受金属退火启发的元启发式优化：以温度 T 控制接受劣解的概率 P=exp(−ΔE/T)，温度从高到低退火，先在解空间大范围搜索（接受劣解逃出局部最优）、后收敛到近似最优。空间映射（spatial mapping）把 DFG 节点放置（placement）到 CGRA 的 PE 阵列并路由（routing）互连，是 CGRA 编译的关键 NP-hard 问题。
- 在 LoRA 中的作用：后端工具用 SA 空间映射把 DFG 映射到 6×6 CGRA（36 PE + 12 IOB）；同时用内存分区算法（借鉴 [15][17][74] 的 constraint satisfaction/graph coloring）把数据分配到多 bank SPM 并把 bank 冲突访问调度到不同时间槽防止内存争用。论文称"类似 [16][71] 的 simulated annealing-based spatial mapping"（COFFA 框架继承），未给出 SA 超参细节。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（LoRA 后端）：DFG（含 XCore 自定义节点）→ SA 映射：随机初始放置 → 每轮以温度 T 扰动（移动/交换节点位置、重连路由）→ 评估代价（时延/拥塞/能否布线，含 XCore 节点需落位到 XCore-PE 位置、GIB 互连资源）→ 按 Metropolis 准则接受/拒绝 → T 按退火表下降 → 终止后输出最优配置（各 FU 配置字、GIB 布线配置、IOB 地址生成配置）→ 生成 CGRA calling function（load 配置到 SPM、配置 CGRA、激活执行、写回）。
- 例：Swiglu 的 15 节点 DFG（含 1 个 XCore 节点）映射到 6×6 CGRA（2 个 XCore-PE 位置），映射成功与否决定 CGRA 尺寸选择（论文按最大节点数+后端可映射性设定 CGRA 尺寸）；内存分区算法同时决定 12 个 SPM bank 的数据布局与冲突调度。
- 其它用法（vault 中广泛出现）：RSU 论文基于开源 CGRA 软件栈做 SA 静态映射优化（把一步搜索拆成三部分：叶节点距离计算等）；Crane/SET 用 SA 做层间调度/数据复用；Efficient Multimodal Serving 用 SA 与 GA/贪心对比搜索配置。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：LoRA 的 FGRA-Compiler（后端，C++，开源于 COFFA 仓库 LoRA-ISCA-AE）；典型 SA 参数含初始温度、退火率、每温度迭代数与终止条件（论文未明确给出，写"论文未明确说明"）。通用实现可用库（如 SimulatedAnnealing.py）或自研。
- 使用场景：CGRA/FPGA 的放置布线、调度优化、配置搜索等组合优化；与遗传算法（见算法pipeline条目）同为采样式启发式，常被论文对比（SA 通常更稳定、GA 有时更快/更好）。局限：解空间大时收敛慢、需多次运行取优；LoRA 依赖后端能否找到映射解来决定 CGRA 尺寸（公平性前提）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）
