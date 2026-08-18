## gem5 Garnet / HeteroGarnet（chiplet NoC 模拟基础设施）

术语解释
gem5 是模块化全系统架构模拟器；Garnet 是其 Ruby 内存系统内的逐周期 NoC 模型；HeteroGarnet（Garnet 3.0）是面向 2.5D chiplet/异构互联的扩展。DICE 在该体系内新增 PHY 建模。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
gem5 支持 SE（系统仿真）与 FS（全系统，引导 Linux）两种模式，组件包括 OoO CPU、缓存层次、Ruby（缓存一致性内存系统，协议如 MOESI/MESI Two-Level）与网络模型。Garnet 2.0 是 Ruby 内的逐周期 NoC 模型：GarnetNetwork + NetworkInterface（连接一致性控制器与路由器）+ Router（仲裁与流控）+ NetworkLink + CreditLink（credit 背压），flit 默认 16B，支持任意 Ruby 拓扑（mesh 等）。HeteroGarnet 为其 chiplet 扩展：时钟域岛 + CDC FIFO（DVFS 建模）、SerDes 单元（变带宽链路）、多端口 NI；论文（Kite，DAC 2020）以 2.5D 异构 interposer 拓扑为对象，用带宽限流模拟 SerDes——但无 PHY/噪声/FEC 建模，这正是 DICE 的切入点（Web 证据：gem5 官方文档 garnet-2/heterogarnet、Kite 论文）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DICE 对 gem5 的修改（Table II 配置）：CPU 侧 3.0 GHz x86_64 OoO（ROB 512、IQ 160、LQ/SQ 160）→ Ruby MESI-Two-Level → Garnet CCD 内 2×4 mesh（2.0 GHz、128-bit、1-cycle router/2-cycle link）→ chiplet 边界 PHY 路由器（DICE 新增：FEC/调制/噪声/LLR/解码/ACK-NACK）→ IOD 2×2 mesh（1.0 GHz，8 内存控制器，32GB DDR5 4400）。模拟器输入→性能输出：SE 模式加载 14 个 benchmark（每 CCD 一进程）或 FS 模式跑 Linux+C2C benchmark → 逐周期推进 CPU/缓存/网络状态机 → PHY 阶段逐 flit 计算编码/噪声/解码延迟与重传 → 统计每包延迟分解、尾延迟直方图、IPC/执行时间；验证输出为对真实 AMD EPYC 9454P 等 C2C 延迟的 RMSE（HG 141.2→DICE 89.5 cycles）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：gem5 用 Python 配置（模拟系统布局，如 CCD/IOD 拓扑）+ C++ 时序模型（路由器/链路/PHY）；HeteroGarnet 在 Ruby 拓扑层加 ExternalLink/InternalLink 与时钟域支持。使用方式：chiplet 研究以 gem5+HeteroGarnet 为默认基线（固定延迟+限流），DICE 在此基础上把 PHY 阶段替换为运行时模型；DICE 代码开源（GitHub RashidAGP/DICE-Simulator、Zenodo 19428665，ISCA'26 三枚 artifact 徽章）。开销：相对 HG 运行时 +0.3–26.1%（平均 9.2%），大头在 FEC 解码，可用 memoization 优化。

Omelet 补充视角（ISCA'26，2.5D/3D chiplet 分层互连模拟器）：Omelet 直接集成修改 gem5 v21.1 的 Garnet NoC 模拟器，将其周期级仿真扩展为异构、层次化、封装感知的 2.5D/3D 互连。与 DICE（在 Garnet 中加 PHY/FEC 运行时建模）不同，Omelet 的修改重点是：(1) 把单一 on-chip 网络扩展为 NoC/NoI/NoL 三层网络图（Network-on-X），router 间连接按 3D 放置坐标 x,y,z 自动判 2.5D lateral / 3D vertical 并打技术标签；(2) 每条链路挂技术感知元组 ⟨W flits/cycle, t_cyc cycles, E_bit⟩（来自 HFSS+SPICE 离线表征的技术表）；(3) 检测技术域失配时自动插入 PHY transition queuing adapter、SerDes 序列化/聚合延迟（∝数据率比）与 CDC（固定 1 cycle）；(4) router 按连接链路最低带宽 dimension、buffer 深度按 delay-bandwidth product 自动配置（NoC 4 flits/VC、NoI 8 flits/VC）。运行于 RHEL 7.9 / Intel Xeon Gold 6226R 64 核 / 502 GiB RAM，单核每 run，12 chiplet 全系统约 1216.5s；还与 gem5 Full-System 集成（KVM 引导→Timing CPU，PARSEC 流量注入，FS 比合成流量慢 42×–3126×）。

涉及论文标题：
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
