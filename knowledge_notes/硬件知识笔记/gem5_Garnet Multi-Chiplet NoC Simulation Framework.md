## gem5/Garnet Multi-Chiplet NoC Simulation Framework

术语是什么？

gem5 是一个开源的全系统计算机架构模拟器（https://github.com/gem5/gem5），支持多种 CPU 模型（out-of-order、in-order）、cache 层次结构、coherence 协议和 memory 系统。Garnet 是 gem5 中 Ruby memory system 的详细 NoC 模型，提供 cycle-accurate 的 router、link、VC 和 flow control 模拟。HeteroGarnet 是 Garnet 的扩展，支持异构 chiplet 和多 die 间的 NoC 模拟。本文使用 gem5 + Garnet 构建 4-chiplet × shared interposer 多 chiplet 模拟环境（chiplet 和 interposer 均为 4×4 mesh、XY routing、3 VN、per-VN 2/4 VC、virtual cut-through flow control），评估 DFBM vs MTR/DeFT/RC 的 latency、throughput 和 PARSEC full-system speedup。

从硬件架构角度拆解术语：

gem5/Garnet 的模拟层次：(1) gem5 CPU models (x86 out-of-order cores) 执行 PARSEC application → 产生 memory/cache request；(2) Ruby memory system (MESI Two Level coherence protocol) 将 request 翻译为 coherence transaction → 注入 NoC；(3) Garnet NoC model 计算 cycle-accurate 的 router pipeline delay (RC/VA/SA/ST/LT)、link traversal delay、VC allocation 和 credit-based flow control；(4) 论文修改 Garnet 以实现 DFBM 的 CM credit tracking（Expected Credit Table lookup + pre-allocated/reserved credit 管理 + CVN-DB occupancy 查询）和 CVN-DB 的优先级入队/出队逻辑，以及 MTR/DeFT/RC 三种 baseline 的边界机制。Synthetic traffic (Uniform-Random/Transpose/Bit-Rotation) 通过 Ruby's synthetic traffic generator 注入，绕过 CPU model 直接测量 NoC-level latency 和 saturation throughput。Full-system 评估启动 x86 Linux + PARSEC benchmark suite，测量 application execution time（真实性更高但模拟速度慢 100-1000×）。

术语一般如何实现？如何使用？

gem5 使用 C++ 实现，Python 用于系统配置脚本。典型使用流程：(1) 编写 Python 配置脚本定义 chiplet 数量、NoC topology (mesh/torus/cmesh/etc.)、routing algorithm、VC 数、buffer depth、flow control 模式；(2) 定义 CPU type、cache hierarchy、coherence protocol；(3) 对 synthetic traffic，设置 traffic pattern 和 injection rate sweep；(4) 编译 gem5 并运行：`build/Garnet_standalone/gem5.opt configs/example/ruby_network_test.py`（synthetic）或 `build/X86_MOESI_hammer/gem5.opt configs/example/fs.py`（full-system）；(5) 解析 gem5 stats.txt 输出提取 latency、throughput、buffer occupancy 等指标。在本文中，为模拟 multi-chiplet，还需配置 interposer 的 vertical channel 数量（uniform 16 条 / non-uniform 12 条分布不均）以评估 floorplan 约束下的鲁棒性。论文的 DFBM 修改未开源，但 Garnet 本身开源且支持插件式 router/flow control 扩展。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

