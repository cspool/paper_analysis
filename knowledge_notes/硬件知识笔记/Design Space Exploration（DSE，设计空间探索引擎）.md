## Design Space Exploration（DSE，设计空间探索引擎）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DSE 是把模拟器变成自动化探索工具的一类引擎：给定设计变量（如拓扑、路由器设计、chiplet 放置、封装选项）的搜索空间，自动对每个设计点运行完整仿真流程并收集性能结果，输出 Pareto 最优设计集。Omelet（ISCA'26）的 DSE 引擎把设计点形式化为 ⟨G_net, P_place, T_tech⟩（NoC/NoI/NoL 拓扑与 router 参数、chiplet 坐标与 tier 分配、interposer 材料/bonding/TSV pitch 等），用户给出离散集或范围后引擎形成笛卡尔积。评估循环每点四步：生成技术感知链路 → 合成 NoX 网络 → flit 级周期仿真 → 输出平均 flit latency、峰值吞吐、per-link EPB 与 traffic-activated energy、per-link utilization、crossing overhead。瓶颈归因：仿真器给每条链路/路由器挂物理与架构元数据，延迟升高时输出 latency breakdown 定位具体原因。搜索策略：默认穷举 ≤10³ 点，更大空间用加权目标引导的模拟退火。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Omelet DSE 的运转流程：定义搜索空间（如 7 种 NoI 拓扑 × 2.5D/3D 集成方案 × 硅/有机 interposer 材料）→ 逐设计点跑完整 5 阶段流程（配置→链路建模→放置→NoX 仿真）→ 收集 4 指标（延迟 L、峰值吞吐 T、总功率 P、最坏链路利用率 U）→ 输出 Pareto 前沿。使用例子（Fig.13）：7 拓扑 × 2 集成方案（2.5D/3D）× 2 interposer 材料的 radial 图——硅 interposer 下 mesh 达到最优延迟/吞吐（高布线密度多短路径、path diversity 高）但功率/利用率更高；有机下 DoubleButterfly 与 Kite 家族更优（少 hop 高 reach 直连，减少穿多个小链路）。结论：拓扑最优性依赖封装技术，DSE 必须联合拓扑-网络-集成一起搜（同一设计在不同技术下不再最优）。鲁棒性验证：±20% latency 扰动下拓扑排序在 50ps 基线稳定（10ps 小基线会更敏感）；latency(α) 与 EPB(β) 0.8×–1.2× 扰动下 394 个设计点的 Pareto 前沿近均匀平移、不改变 knee 点与最优选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Omelet 的 DSE 引擎复用同一 NoX 仿真流程，输出附设计点对应的 placement/network 配置文件；当前评估性能与能量，可通过外部模型扩展成本/热等指标。相关工具背景：既有 chiplet DSE 工具（RapidChiplet 等）用解析式（快但丢失技术细节，<13s 级运行），Omelet 用周期级仿真换取技术保真（12 chiplet 约 1216.5s），故 Omelet 以合成流量注入做早期 DSE、以 gem5 Full-System + PARSEC 验证选点。使用方式：配置搜索空间 → 跑 DSE（穷举或模拟退火）→ 输出 Pareto 最优设计集 + 每点指标 + 瓶颈归因，供架构师在 topology/placement/packaging 联合维度上做 co-design 决策。

RHODES 的碳感知 DSE 视角（ISCA'26）：RHODES 是另一类 DSE 引擎——不以模拟器为评估器，而以"工作量 profiling 数据 + 鲁棒 MILP 优化 + Monte Carlo 验证"为评估循环。设计空间是异构 SoC 配置：CPU 配置集 C={1..32 核}（面积按 AMD EPYC 7453 核 10.125 mm²@7nm EUV + 每核 6.5 mm²@12nm I/O die）、GPU 配置集 G={m∈1..128 SM × 11 档频率 210–765 MHz，默认 765MHz}（SM 面积按 NVIDIA A100 826 mm²/108 SM=7.65 mm²@7nm）、可选 HBM3（8 die 16GB、107.09 mm²、7 pJ/bit）与 DSA（4× GPU 效率，用于 WLP 分析）。评估指标：执行时间（工作量 setup/compute/teardown 三阶段）、功耗（TDP 估计）、面积、成本、总碳 tC；约束包括 T_max/P_max/A_max/tC_max。运转流程：输入 HILP profiling 数据（Rodinia 负载）→ 编码不确定碳参数进鲁棒 MILP → Gurobi 求解（45,088 配置 <5 分钟、0 间隙）→ 输出对不确定性集所有实现可行的最优配置 → Monte Carlo（2000 样本/参数、5 种分布）验证 p95 tC 与违反率。与 Omelet 的周期级仿真 DSE 相比，RHODES 用解析 profiling + 数学优化换取早期设计阶段在"碳×不确定性"维度的高通量探索（nominal 设计在真实变异下违反 tC 约束 49.85–54.4%，RHODES 降到 ≤1.1%）。三种目标：min tC（给定性能约束）、min T（给定 tC 约束）、min tCDP（Pareto 追踪线性化）。

涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
