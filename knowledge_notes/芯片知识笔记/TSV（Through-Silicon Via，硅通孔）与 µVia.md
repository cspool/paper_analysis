## TSV（Through-Silicon Via，硅通孔）与 µVia

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TSV 是穿透硅片/硅中介层的垂直导电通道（深孔填充金属），用于 3D 堆叠 die 间或 die 到 interposer 的垂直互连；µVia 是介电层/有机材料中的微型垂直通道。TSV 的几何参数为 pitch（间距，决定垂直互连密度）与 aspect ratio（深宽比，AR，决定制造难度与寄生）。相关笔记：knowledge_notes/芯片知识笔记/Cu-Cu Hybrid Bonding（对比：hybrid bonding 在表面直接形成 Cu 触点，无需 TSV 钻孔；HBM 的 TSV pitch ≈10µm，hybrid bonding 1µm → 垂直密度提升约 25×）、HBM Logic Die（HBM 堆叠中 TSV 连接 DRAM die 与 logic die）。Omelet（ISCA'26）把 TSV/µVia 作为 die-to-die 链路组件链中的 lumped 元素（贡献固定延迟，不随距离缩放），参数：TSV pitch=同 bonding pitch、AR=15；µVia pitch=同 bonding pitch、diam=pitch/2；并做物理约束检查（TSV 密度 per mm²、垂直堆叠重叠区限 TSV/hybrid bond 数量）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
TSV 决定 3D 垂直互连（NoL）的密度与带宽。Omelet 的建模流程：HFSS 提取 TSV/bonding tier 的 RC 寄生（单信号线 + 双相邻地线结构）→ 技术表 → 垂直链路带宽按 λ=重叠区可布 TSV 数/bump pitch 计算 W=floor(λ·R_lane/(f_clk·F_bits))。执行例子（3D 2-tier 系统）：上层 chiplet 与下层 chiplet 的重叠区域允许 TSV 形成垂直链路 → NoX 引擎按 x,y,z 坐标把跨 tier router 连接标为 vertical link → 查技术表得 t_cyc 与 W（hybrid bond 1µm pitch → λ 大、垂直带宽高；solder ball 30µm → λ 小、带宽低）→ 垂直链路参与周期级仿真。F2F/F2B 方向建模：3D 垂直路径由 bonding 组件按 F2F（face-to-face）或 F2B（face-to-back）链式组合，区别在 TSV 于 bonding 界面前/后穿过 die。Omelet 的物理约束：TSV 密度 per mm²、interposer 最大 span（硅不可 span 20mm）、beach-front 共享周长限制横向链路数、垂直重叠区限制 TSV 数——配置不可行时报错。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TSV 制造为深反应离子刻蚀（DRIE）打孔 → 绝缘/阻挡层 → 电镀铜填充 → CMP；µVia 在 RDL/介质层中形成（如有机 interposer 下沉到硅桥的 µvia 路径）。使用方式（Omelet）：用户在 technology library 配置 vias 选项（TSV pitch=same as bond、AR=15；µVia pitch=same as bond、diam=pitch/2），Omelet 据此构建 2.5D/3D 垂直链路并参与周期级仿真；新封装技术只需在 per-link 参数表加条目 + 配置文件引用即可扩展（如把被动 interposer 扩展为 silicon-bridge：在 2.5D 链中插入被有机材料包围的 µvia）。作用：把 TSV 几何（pitch/AR/密度）对系统级垂直带宽、延迟与可行性的影响纳入架构级评估，支撑 3D 集成（Foveros、X-Cube、混合键合）的设计探索。

  - SHyLA 补充（3D 堆叠混合内存的 TSV）：SHyLA 采用 TSV pitch 10μm（按制造实践 [58]）、TSV 能量 0.167pJ/bit；3D 堆叠经密集 TSV 实现二维 areal bandwidth 缩放（带宽随 die 面积缩放，区别于 2D/2.5D 沿封装边缘 PHY 的一维缩放），使设计者能把 NVM 的 cell 面积换成 TSV/外围电路（decoder、row buffer）提带宽而不显著损容量。DRAM/NVM 采用 4-Hi 堆叠并行放置于计算 die 上方、信号经 buffer die 重路由（避免垂直堆叠共享 TSV 网格的工艺对齐约束）；CACTI-3DD 按此 TSV/堆叠结构推导带宽与容量。
涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
