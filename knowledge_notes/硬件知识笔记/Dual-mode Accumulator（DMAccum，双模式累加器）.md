## Dual-mode Accumulator（DMAccum，双模式累加器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DMAccum（Dual-mode Accumulator）是 HiT 的双模式部分和（psum）累加器，处理两类输出布局：(1) HS×HS 输出高度稀疏且分布不规则，psum 以压缩格式存于 Local Buffer（每行打包多个稀疏值），新 psum 到达时需要先找列索引匹配位置；(2) 其他稀疏工作负载输出密度高，直接按行列索引稠密寻址更新。两种模式切换由输出密度决定，是"利用输出稀疏性"的硬件落地。外积数据流的普遍痛点正是 psum 量大且不规则——Web sources 显示 IOPS（IEEE TC'25）用 address mapping 累积不规则稀疏 psum、Sparm（ASAP'24）与 SpMARD（TACO）分别用 indexing unit 与 Position-Based Psum Array（PPA，VID 直接映射地址）解决同一问题。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
HiT 每 Compute Group 的 DMAccum 含 32 个 FP32 加法器与 512 个比较器。HS×HS 模式采用 binning-compare-update 流程：(1) 把到达的 psum 按列索引做 modulo 运算映射到内存 bin，bin 索引+行索引确定目标 bank 与行；(2) 每个 bin 只装 2 项，psum 只需与 2 个候选比较而非整行（减少比较器数）；(3) 三种结果之一——匹配则更新该条目、无匹配但有空槽则插入（冲突用 priority encoding）、bin 满则 spill 到 overflow buffer 下轮再累积；(4) 当一行所有 bank 都满且新 psum 映射到该行时，整行 row-granularity spill 到 Global Memory、清空后继续，tile 完成后再第二遍合并（平均仅 3.30% 输出行 spill、4.05% 开销）。其他模式直接行列索引累加。压缩格式使 HS 输出大矩阵也能用小 Local Buffer 装下，从而允许更大的输入 tile（更多非零、更高交叠率与计算密度）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：register file 型 Local Buffer（每 bank 4 读 4 写，multi-ported，低延迟支撑高速 psum 访问）+ binning 比较逻辑 + overflow 缓冲；HiT 论证了为什么用 register file 而非 SRAM——CACTI 建模同等带宽的 multi-ported SRAM 面积是 register file 方案的 16.7×。使用：HSparse 时 DMAccum 接收来自本 Row 或经 ring network 路由来的 psum；MSparse 时乘法器直接送 DMAccum（bypass PSum Router）；D×D 时 DMAccum 退化为普通加法器（线性加法网络）参与 systolic 垂直累积，sparsity 逻辑 clock-gated。评估指标：HS 模式下 DMAccum 活跃占比高、stall 低于 1% 周期。论文未开源。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
