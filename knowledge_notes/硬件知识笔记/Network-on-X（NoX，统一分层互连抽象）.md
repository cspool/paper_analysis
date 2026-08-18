## Network-on-X（NoX，统一分层互连抽象）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Network-on-X（NoX）是 Omelet（ISCA'26）提出的统一网络抽象：把片内 NoC、interposer 级 NoI、垂直层间 NoL 三种物理与电学特性截然不同的互连域整合进同一个 flit-level cycle-level 网络图，使跨层流量、边界效应与 backpressure 在单一仿真内核中自然传播，而非像既有模拟器那样逐域独立仿真再事后相加延迟。NoX 引擎通过五个自动化阶段把物理链路数据与放置坐标变换为可仿真的 flit 级网络图：(1) 系统与节点实例化（计算节点、内存控制器、indirect router 经 external link 连路由器）；(2) 放置感知链路构建（按 router 对 x,y,z 坐标判 lateral/vertical 并打技术标签：on-die wire / interposer RDL / vertical TSV）；(3) 技术参数转换（物理 Gb/s、ps、pJ/bit → ⟨W flits/cycle, t_cyc cycles, E_bit⟩，W=floor(λ·R_lane(ℓ)/(f_clk·F_bits))，F_bits=128）；(4) 自动 adapter 插入（技术域失配 → PHY queuing adapter + SerDes 聚合/序列化 + CDC 1 cycle）；(5) 路由合成（按连接链路最低带宽 dimension router、按 delay-bandwidth product 配 buffer、传播 turn 限制等路由约束）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
NoX 是 Omelet 的分层核心机制，直接对应论文贡献 C1（Unified Hierarchical Network Modeling）与 L1 缺陷的修复。运转例子（2.5D/3D 系统一次跨 chiplet+跨层 flit）：core 注入 → NoC router（4 级流水、4 VC、4 flits/VC）→ 边界 adapter（SerDes 延迟 ∝ 数据率比、CDC 1 cycle）→ NoI router 逐 hop（每 hop 延迟/带宽按技术表）→ 若需跨 tier 再经 adapter 上 NoL（垂直 TSV/bonding）→ 目标 NoC → 目标 core。关键效果：NoI 拥塞回压 NoC 注入端（统一仿真下更晚饱和、中等负载端到端延迟显著更低），interposer 饱和时拥塞传播进 NoC/NoL 压缩整个 stack 注入率；isolated 模型（改进版甚至回放 NoC 的精确 ejection 时序给 NoI）无法复现这些行为——unified 与 isolated 的 load-latency 曲线无一致偏移/缩放关系，证明端到端性能不能由逐域仿真线性重构。NoX 是 Omelet 把"组件级物理链路建模"接入"周期级架构仿真"的桥梁，也是 DSE 引擎复用同一仿真流程的载体。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 gem5 v21.1 的 Garnet NoC 模拟器扩展（Omelet 修改 Garnet 支持异构、层次化、封装感知互连）；链路参数来自离线 HFSS 3D 电磁提取 + SPICE 电路仿真的技术查找表（用户只需指定集成技术，无需手工调 latency/bandwidth 常数）。使用方式：输入三类配置（technology library、network configuration、system configuration）→ Omelet 顺序执行技术感知链路建模 → 逻辑放置（force-directed 自动放置或用户坐标）→ NoX 构建 + 周期级仿真 → DSE。输出：平均 flit latency、峰值吞吐、per-link EPB 与 traffic-activated energy、per-link utilization、crossing overhead、瓶颈归因（latency breakdown）。作用：让 NoC/NoI/NoL 的交界（PHY/SerDes/CDC 开销、带宽失配、跨层拥塞）成为可量化的一阶效应，支持 2.5D/3D 系统在 topology×placement×packaging 联合空间上的设计探索。

涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
