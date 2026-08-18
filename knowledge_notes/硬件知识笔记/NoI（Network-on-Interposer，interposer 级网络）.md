## NoI（Network-on-Interposer，interposer 级网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NoI 是 2.5D/3D chiplet 系统中承载跨 chiplet 通信的 interposer 级互连网络：信号沿 interposer 上的 RDL（redistribution layer）走线在 chiplet 之间传输，走线可达毫米级、coarse-pitch、高 RC，带宽/延迟/能耗由封装技术决定。Omelet（ISCA'26，Georgia Tech）把 NoI 与片内 NoC、垂直 NoL 统一进单一 flit-level cycle-level 框架（Network-on-X）：NoI 链路参数从 Ansys HFSS 3D 电磁提取 + SPICE 电路仿真的技术表取回（按 interposer 材料、RDL L/S、bonding pitch、线长 ℓ 索引），以元组 ⟨W flits/cycle, t_cyc cycles, E_bit⟩ 进入仿真。NoI 的物理载体包括 passive silicon/organic interposer 与 silicon bridge（EMIB 式嵌入式桥接，Omelet 通过在 2.5D 链中插入被有机材料包围的 µvia 模拟）。相关笔记：knowledge_notes/芯片知识笔记/Active Interposer（主动中介层，interposer 内嵌 NoC 路由器的形态）、CDG for Multi-Chiplet NoC（interposer NoC 通道参与跨 chiplet 死锁环分析）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Omelet 中 NoI 是层次化互连的中层，router 为 indirect router（只转发、不直接接注入/排出节点），chiplet 内部 NoC router 为 direct router。执行流程（一次跨 chiplet flit）：chiplet0 的 NoC 边界 router → chiplet–interposer 边界 adapter（检测 NoC 快窄→NoI 慢宽失配，插入 PHY transition + SerDes 聚合/序列化，延迟 ∝ 带宽比，CDC 1 cycle）→ NoI router 逐 hop 经 interposer RDL 链路传输：每 hop 延迟 t_cyc 与带宽 W=floor(λ·R_lane(ℓ)/(f_clk·F_bits)) 由技术表按长度 ℓ 索引（λ=可用 bonding perimeter/bump pitch；R_lane(ℓ) 随长度 RC 衰减）→ 目标 chiplet adapter 反序列化 → 目标 NoC。NoI 拥塞会跨层回压：interposer 链路饱和时背压传播进 NoC/NoL，压缩注入率（这是统一建模与 isolated 建模的关键差异：isolated 模型把 NoC/NoI/NoL 独立仿真后相加，NoI 饱和不影响 NoC 注入，导致过早饱和与吞吐高估；Omelet 显示 tech-agnostic 配置低估端到端延迟平均 32.9K cycles / 53.5%）。7 种 NoI 拓扑（mesh/cmesh/dblbut/butdon/Kite-S/M/L）× 2.5D/3D × 硅/有机 interposer 的 DSE 显示：拓扑最优性依赖 interposer 材料（硅下 mesh 优、有机下 DoubleButterfly/Kite 优）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NoI 路由器可放在 interposer 上（active interposer，如 RDL 层内嵌 NoC，见 Active Interposer 笔记）或作为 chiplet 内独立 router 经 TSV/µbump 上到 interposer；链路为 RDL 走线 + TSV/via + bonding。Omelet 的使用方式：用户配置 interposer 材料（silicon/organic）、RDL L/S、bonding pitch、NoI 拓扑与 chiplet 放置坐标，Omelet 查技术表生成每链路 ⟨W, t_cyc, E_bit⟩ 并做放置可行性与 beach-front 检查（共享周长限制可布链路数）。仿真用 gem5 Garnet（Omelet 修改扩展）；评估用 5 种合成流量（uniform random/shuffle/transpose/tornado/neighbor）。作用：把 interposer 级通信从"单一抽象链路/常数 hop 代价"升级为"技术感知 + 组件链 + 跨层交互"建模，支撑 architecture–packaging co-design 与 open chiplet 生态（同一 chiplet 部署不同封装平台）的评估。

涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
