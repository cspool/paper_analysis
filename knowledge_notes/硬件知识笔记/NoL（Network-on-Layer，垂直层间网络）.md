## NoL（Network-on-Layer，垂直层间网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NoL 是 3D 堆叠 chiplet 系统中承载 tier 与 tier 之间垂直通信的网络：信号通过 TSV 或 bonding 技术（micro-bump、Cu-Cu TCB、hybrid bonding，F2F/F2B 方向）在堆叠层之间传输。与 NoI（横向 interposer RDL，毫米级、高 RC）不同，NoL 链路短、低延迟、高密度（细 pitch），但带宽/密度受 bonding pitch、TSV 几何（pitch、aspect ratio）与垂直重叠面积约束。Omelet（ISCA'26）将 NoL 作为 Network-on-X 统一分层互连的最高层，垂直链路按 bonding 组件链建模（F2F 与 F2B 的区别在 TSV 于 bonding 界面前/后穿过 die），链路参数同样来自 HFSS+SPICE 技术表。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Omelet 中 NoL 的运转：NoX 引擎按 router 对的三维坐标（x,y,z）判定——同 tier 为 2.5D lateral link，跨 tier 为 3D vertical link（NoL）；垂直连接只在上下层重叠区域内可形成（TSV/hybrid bond 只能在重叠区），Omelet 的 Placement Engine 计算垂直重叠面积并据此限制链路数。执行例子（3D 2-tier 系统 8 chiplet，NoC→NoL→NoC→NoI→NoC→NoL→NoC 全层次路径）：packet 在下层 chiplet 的 NoC → 边界 adapter 上 NoL（垂直 TSV/bonding，短而快，带宽 W 由 bonding pitch 决定：hybrid bond 1µm → 细 pitch 高密度，solder 30µm → 粗 pitch 低密度）→ 上层 chiplet NoC → 又经 NoI 到另一 stack → ... → 目标。实验发现：虽然垂直链路本身快，但跨 stack 流量（tier 间访问 HBM/集中式 I/O）仍需共享 interposer 路由通道，interposer 链路/路由器成为吞吐限制资源——3D stacking 的系统级延迟最终被共享 interposer 争用主导（Takeaway 5），说明必须联合优化 interposer 带宽、路径多样性与流量放置才能兑现 3D 优势。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：垂直互连的物理形态为 TSV（穿硅通孔，pitch 常与 bonding 一致、AR 15）或 bonding 界面（µbump、Cu-Cu TCB、hybrid bonding）；NoL 网络可每层各自路由器经垂直 link 相连，也可用垂直专用路由器。Omelet 使用方式：用户在 system configuration 指定 stacking depth（1/2/3）与每层 chiplet 布局，NoX 自动构建 NoL 并配置垂直链路参数（router 尺寸与带宽/延迟/序列化开销均可配置）。评估 3D 用 2-tier（8 chiplet）与 3-tier（12 chiplet）配置，每 tier 保持 2×2 物理布局。作用：量化垂直集成（Foveros/AMD 3D V-Cache/Samsung X-Cube 类）在系统级的真实收益与瓶颈，避免"垂直链路快 → 3D 一定好"的错误推断。

涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
