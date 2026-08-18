## 稀疏矩阵乘法数据流（Inner Product / Row-based / Outer Product Dataflow）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
稀疏矩阵乘法 C=A×B 的执行数据流指三个嵌套循环（M/K/N 维）的执行顺序，它同时决定数据复用与索引对齐控制复杂度。三种经典数据流（Harmonia 论文 Fig.2）：(1) Inner Product（InP，内积）——每个输出元素 C_m,n 做行-列点积，提供强输出复用但弱输入复用（B 的每列需反复取）；(2) Outer Product（OutP，外积）——对每个 k，用 A 的一列×B 的一行生成 psum 矩阵（rank-1 update），最大化输入复用但需要大量 psum 归并；(3) Row-based（Row，行式）——A 的非零 A_m,k 与 B 整行 B_k,: 相乘，复用中等、归并开销小。三者互补，没有单一映射能同时高复用低开销——这正是多功能稀疏加速器（Trapezoid 等）存在的动机。Harmonia 的关键发现是：数据流最优性还取决于 tile 形状与 occupancy（K 小 N 大时 OutP 最优，K 大时 InP/Row 受益；tile 形状 (64,128,64) 使 Row 优于 OutP），且会随 tile 稀疏模式变化（email.mtx 上最佳数据流从 OutP 漂移到 Row），因此需要运行时按 tile 选择。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Harmonia 的 32 PE 行架构（每行 32 FP32 乘法器 + radix-16 MRN + 16-to-16 Benes DN + 1KB BUF）用同一数据通路实现三种数据流，仅 DN 路由、MRN 归约模式与 on-row buffer 策略不同，在 tile 边界由 Reconfiguration Engine 重编程：
- InP（Fig.6）：每个 PE 行驻留一个 A 行，对应 B 列片段经 DN 流式广播并沿 PE 行链流水转发给下游行（最大化 B 跨行复用）；每 PE 产生部分积进本行 MRN 的 merge-before-store 模式，先压缩 psum 再缓冲。适合中稀疏、行密度均匀的 tile。
- Row（Fig.7）：每个 PE 行绑定一个 A 行，DN 按该 A 行 nnz 列号选择性路由所需 B 行片段（operand gating），B 片段在 on-row BUF 缓冲一次供本行所有 PE 共享；MRN 沿单一 A 行轨迹做行顺序归约，merge 深度浅且可预测。适合行密度高度不均匀/突发（bursty）的 tile。
- OutP（Fig.8）：DN 把每对 (A_*k, B_k*) 广播到所有行（rank-1 更新流），MRN 以 column-accumulate 模式归约，merge 深度极浅、psum 缓冲需求最小、分发压力最小。适合高稀疏/强聚类拓扑。
三种模式切换 = pipeline flush + DN/MRN 重编程 + buffer 重置，20–50 cycles，总 stall <1%。数据格式按 tile 稀疏度选：高稀疏用 CSR/CSC+坐标列表，轻度稀疏用 bitmask。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在硬件上，数据流本质是"操作数路由 + 部分和归约路径"的静态/可重构配置：固定数据流加速器（SIGMA 的 Flex-DPE、HighLight）把 PE 微架构绑死假设的稀疏模式，模式偏离时利用率崩（SIGMA 高稀疏下 <10%）；灵活加速器（Flexagon）用重路由/缓冲/控制支持多模板但 75% 面积是非计算结构；Harmonia 沿用 Trapezoid 的折中：同构 PE 阵列 + 仅重配置 DN 与 MRN 两个部件，接近常效。论文用 cycle-accurate 模拟器（建模 MAC、MRN、DN、本地 buffer、SRAM、HBM）验证三种数据流在 16 个 SuiteSparse SpMSpM 矩阵与 4 个剪枝 DNN（LLaMA-0.2/0.4/0.6、OPT-0.2/0.4/0.6、ResNet-0.1/0.2、VGG-0.1/0.32）上的行为。论文未提供开源实现。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

HiT 补充视角（ISCA'26，统一稀疏自适应架构下的三类数据流）：HiT 把稀疏矩阵乘法数据流从"三种经典模板"扩展到"HSparse / MSparse / 内积"三态，核心判断是数据流选择必须同时解决两个可扩展性瓶颈——Gustavson 的随机访存与朴素外积的 psum 爆炸：
- Gustavson（row-stationary，Trapezoid 采用）在大规模并行下对 Matrix B 产生并发随机访问：128×128 MAC 分 4 cluster、每 cluster 32 PE 行共享 4MB 32-bank cache 经 32×32 crossbar，HS×HS 时并发请求频繁命中同一 bank（图 5：两个 PE 都访问 Bank 1 只执行 1/4 乘法），HS×HS 仅达峰值吞吐 3.125%（512/16K MAC/cycle）、256 行时 MAC 利用率降至 ~10%——内存子系统而非计算成为瓶颈，增 bank 数又因 crossbar 面积二次增长而不可行。
- 外积（OP）消除随机访存（A 按列处理、B 行顺序流式），但每个 A 非零生成整行 psum，未合并 psum 随活跃 PE 行数超线性增长（图 4b），朴素实现需 off-chip 累积（OuterSPACE）或大规模 merge tree（SpArch）。
- HiT 的解法：HSparse（HS 工作负载）——外积 + PIDU 两层空间并行（列内多非零 + 列间多列，每周期 4 个 A × 64 个 B）+ PSum Router/ring network 片上路由归约 + DMAccum 压缩格式累积，兼得"顺序访存 + 片上 psum 合并"；MSparse（MS 工作负载）——A 行映射进单个 Row 使 psum 立即本地累积（绕开环网），B 经 cluster-local broadcast 广播共享；D×D 用内积数据流跑 128×128 systolic array。数据流按密度启发式（<10% HS、10-90% MS、>90% D）静态选择，重配置固定周期（占执行时间 0.009% geomean）。评估证明：HS×HS 的 performance/area 比 Trapezoid 高 3.24×、全谱高 1.93×。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

SegFold 补充视角（ISCA'26，动态数据流扩展三类静态模板）：SegFold 指出上述 InP/OutP/Gustavson 分类忽略了一个关键维度——dynamism（动态性），并据此提出动态数据流 Segment：(1) 动态调度（SELECTA）——不再固定循环序，而是在覆盖 K 的 active window（32 个 k）内逐周期贪心选择 (m,k) 对：优先共享 k 以最大化 B 行复用（Gustavson/OutP 的好处），同时避免同 m 不同 k 的对（防止 C 行约简冲突），完成交集的 k 即时退休换新（k 级流水），并支持 partial B 行交错——同时在单 tile 内获得 element-wise A 复用、row-wise B 复用、tensor-wise C 复用；(2) 动态映射（SEGMENTBC）——在虚拟坐标空间 V 中按列序即时定位/创建 C 部分和，merge network 允许元素在 PE 间动态迁移（转发/插入/累加），按运行期 V 空间状态重分配部分和以平衡负载。Flexagon/Trapezoid/Spada 的"多数据流选择"仍停留在静态模板层面（tile 内固定循环序），SegFold 把调度细化到 sub-tile 周期级。结果：15 个 SuiteSparse 矩阵 geomean 1.95× over Spada、5.3× over 最佳 Flexagon 配置；消融证明动态 k 重排（固定 k 序→0.670±0.065）与动态映射（LUT vs zero-offset 1.20×、距 oracle 仅 1.2%）各自贡献。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
- SegFold: Accelerating Sparse GEMM with a Fine-Grained Dynamic Dataflow
