## RDL（Redistribution Layer，重分布层）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RDL 是先进封装中的扇出布线层：在芯片/中介层表面用再分布金属走线把 I/O 从 die 边缘或 bump 阵列重新分布到需要的位置，是 2.5D interposer 与 fan-out 封装（如 TSMC InFO）里 chiplet 间信号的物理载体。RDL 的关键参数为线宽/线距（L/S）、材料（铜 + 介质：oxide 或 polyimide）与层数，直接决定布线密度（可布信号数）、信号衰减与延迟。Omelet（ISCA'26）把 RDL 作为 die-to-die 链路组件链的核心分布式元素：RDL 走线长度相关、按分布式 RC 处理（Elmore 延迟随几何距离缩放），参数范围覆盖 TSMC CoWoS-S 5th Gen（0.8µm L/S，2021）到 CoWoS-L/R（2.0µm L/S，NVIDIA Blackwell B200 2024-2025），并支持前向 0.5µm 与 1.4µm L/S（CoWoS-R 论文数据）；材料选项 Cu/oxide 与 Cu/polyimide（polyimide 6µm 厚）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
RDL 决定 2.5D 互连的物理带宽与延迟上限。Omelet 的建模流程：HFSS 3D 电磁仿真提取 RDL 走线（单信号线 + 双同面地线 + 上下地平面的 GSG 结构）的 RC 寄生，0.1–7GHz、按 pitch 重复提取保持 realistic aspect ratio 与 design rules → 技术表按 interconnect type/material/pitch/frequency 索引 → 仿真时按线长 ℓ 取 R_lane(ℓ)（RC 衰减后的 lane 数据率）计算 NoI 链路带宽 W=floor(λ·R_lane(ℓ)/(f_clk·F_bits))。验证例子：有机 RDL interposer 复现 TSMC 的 RC scaling（1µm Cu 厚、6µm polyimide 介质、2µm L/S），平均差 3.5%、最大 5.7% @7mm。RDL L/S 与 interposer 材料的耦合：硅 interposer 支持细 pitch RDL（0.5µm）→ 更多并行链路（path diversity 高）但单链路延迟/能量高；有机 interposer 用粗 pitch（2µm）→ 链路数受限但单链路延迟低（图 13 的拓扑-材料联合 DSE）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RDL 在 interposer/package 表面逐层沉积金属（电镀铜）+ 介质（聚酰亚胺或 SiO₂），经光刻形成 L/S 走线；fan-out 封装（InFO）用 RDL 替代 interposer 做扇出。使用方式（Omelet）：用户在 technology library 指定 RDL 材料（Cu/oxide、Cu/polyimide）与 L/S、线长范围（0.5–5mm），Omelet 从技术表取回链路参数做周期级仿真；支持"架构驱动"（选预设技术套餐）与"封装驱动"（直接给数值参数探索新兴技术）双接口，并施加技术兼容性 guardrail（如 hybrid bonding 不能配有机 interposer）。作用：使 RDL 的工艺参数（L/S、材料、长度）成为可评估的系统级性能变量，连接封装制造（CoWoS 系列）与架构设计。

涉及论文标题：
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
