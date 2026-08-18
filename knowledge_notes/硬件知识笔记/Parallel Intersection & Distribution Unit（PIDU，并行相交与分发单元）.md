## Parallel Intersection & Distribution Unit（PIDU，并行相交与分发单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIDU（Parallel Intersection & Distribution Unit）是 HiT 加速器中执行稀疏非零索引匹配（intersection）并把匹配对分发到乘法器的硬件单元。"intersection"指把稀疏矩阵 A 中非零元素的列索引与稀疏矩阵 B 中非零元素的行索引做匹配：索引相等才产生有效乘法，否则是无效运算（乘零）。PIDU 的目标是在一次周期内并行匹配多个 A 元素与多个 B 元素（HiT 每 Compute Row 每周期匹配 4 个 A 元素与最多 64 个 B 元素），以低硬件开销替代 Trapezoid 中功耗密集的 intersection unit 与 32×32 crossbar 路由网络。Web sources 确认该概念在文献中普遍存在：FLAASH（arXiv:2404.16317）的 intersection & MAC unit、SPARCAM（多端口动态 CAM 做索引比较）、Google 稀疏点积专利（comparator + matched index queue）都做同一件事——先匹配索引再乘法，避免对零做无效计算。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PIDU 在 HiT 每个 Compute Group（每 Group 32 个乘法器）内工作，4 级流水：(1) 并行列-行索引比较——用 64 个比较器把 1 个 A 元素的列索引与一组（最多 64 个）B 元素的行索引同时比较；(2) leading-zero count 统计匹配位置；(3) 轻量 shifter（非 barrel shifter）按计数把匹配的 B 元素左移对齐到可用乘法器（例如示例中 B 条目左移 1 位后值 2/3 分给 Mult1/Mult2）；(4) 分发到乘法器。匹配数超过 32 时暂停读取新 A 元素直到处理完。因 HS×HS 交叠率极低（geomean 仅 0.12% 的比较产生有效匹配），匹配数极少超过 32。HiT 每 Compute Row 的 PIDU 规格为 4×(64 comparator & shifter)。它对应解决 Trapezoid 的问题：Trapezoid 依赖功耗密集 intersection unit + crossbar 分发网络（crossbar 面积随 bank/PE 数二次增长），PIDU 用规则比较器阵列 + 移位对齐实现大规模低开销匹配，使稀疏执行吞吐能随计算并行度扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上 PIDU 是纯组合/流水化的比较器阵列（64 个 comparator 为一组，每 Compute Group 一组）+ leading-zero counter + 移位器，面积与功耗远低于 CAM 或 crossbar 方案（对比 SPARCAM 的 eDRAM 动态 CAM、Trapezoid 的 crossbar）。使用：在 HSparse（HS×HS/HS×MS/HS×D）与 MSparse（MS×MS/MS×D）两种外积数据流中都被复用——HSparse 中匹配对象来自各 Row 专属 Global Memory bank 流式读入的 B 组；MSparse 中 B 经 cluster-local broadcast 广播到 Cluster 内所有 Row。D×D 模式下 PIDU 被 bypass（数据直送乘法器）并时钟门控以省功耗。HiT 用 Verilog 实现 + Synopsys Design Compiler 22nm 综合评估面积功耗，cycle-level 模拟器逐周期建模其比较/移位/分发行为。论文未开源（NUS，ISCA'26）。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
