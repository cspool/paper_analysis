## Index-to-PE Mapper（IPM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Index-to-PE Mapper（IPM）是 SegFold 的硬件查找表（LUT），为每个到达的 B 元素确定其在 merge network 中的注入起始位置（segment 起点 f_tin），目标是同时保证"合法注入点"（注入点左侧所有 C 列索引 c < b，见 Fig.6d 的禁止场景）与"最小化遍历距离"。因为 V space 中 C 列索引行内单调有序、无空隙（row saturation + column ordering），合法注入点的右边界可由二叉搜索确定——IPM 用树形 LUT 实现 O(log P) 的二叉搜索（P=每行 PE 数），图 7 例子：列索引 11 首层大于 9 走右支、后两层 key 为 null 走左支、到达 leaf=8，即注入到第 8 个 merger。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：B 行进入时只对"行内第一个非零元素"计算映射（row-wise mapping）——因为 row 映射下每个元素起点随列索引同步 +1，结合列序单调性，首元素合法则其余元素自动合法，避免对整行每个元素都查 IPM（否则需与 PE 数成比例的并行读端口）。IPM 更新：PE 更新其 c 值时通知 IPM；LUT 写端口有限，多更新排队串行应用，故 IPM 可能滞后于最新 c 值；由于 time-ascending 性质，过时 LUT 只会把元素映射到真实最右合法点的左侧，不破坏正确性（merge network 仍会纠正），但可能加长 segment。每层 LUT 独立流水（pipelined）以隐藏延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：IPM 是 per-PE-row 的树形 LUT bank（Table II：每行硬件含 shifter、spad bank、LUT bank），深度 O(log P)、总存储 O(P) 每行；与 merge network 保持同步更新（PE 通知 + 串行写队列）。使用：SegFold 用 LUT-based 映射对比 zero-offset（B 行头总是映射到 PE 行起点，f_tin=0）与 ideal-network（oracle 最优放置、无 stale-index 开销）：16 个 SuiteSparse 矩阵上 LUT 映射 geomean 1.20× over zero-offset，相对 oracle 仅 1.2% 平均开销；可扩展性上阵列从 P 扩到 2P 只翻倍 merge network 宽度与 IPM 大小，控制逻辑渐近复杂度不变。

涉及论文标题：
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
