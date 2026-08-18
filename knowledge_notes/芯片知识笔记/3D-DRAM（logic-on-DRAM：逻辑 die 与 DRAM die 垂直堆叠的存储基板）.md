## 3D-DRAM（logic-on-DRAM：逻辑 die 与 DRAM die 垂直堆叠的存储基板）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
3D-DRAM 指把 DRAM die 直接放在逻辑 die 上方/下方、通过密集垂直链路（TSV/µbump）通信的存储基板，与 2.5D HBM（DRAM 堆叠经 interposer 接口）相对。Raptor 采用 logic-on-DRAM 的 face-to-face（F2F）键合：TSMC N4P 逻辑 die 经 36µm-pitch µbump 阵列直接键合到 3D-DRAM die 上，每个 µbump 每周期传 1 bit、单周期宽接口（无 burst、无 DBI pin），短垂直链路把 per-bit I/O 能量压到 0.45 pJ/bit（约 HBM3 的 1/6），同时 100TB/s/card 聚合带宽。3D-DRAM die 组织 840 个 bank（每 bank 1364×124 阵列、row buffer 124 列、每列 32B），映射到 gang/slice 层级形成平衡 channel（每 slice 16、每 chiplet 256）。Raptor 的定位：SRAM 基板（~150TB/s 但仅数 GB 容量）与 HBM（几百 GB 容量但 <20TB/s、I/O 功耗受限）之间的折中——SRAM 级带宽/局部性 + HBM 级容量，更低能量。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 Raptor 芯片设计中，3D-DRAM 不是外挂设备而是原生内存子系统：DRAM die 的 840 bank 与逻辑 die 的 slice/gang/chiplet 层级一一对应（每 slice 16 channel 喂 16 WB），stream blocking 按 bank-group 形成 128B flit channel，stream flipping 在 F2F µbump 接口上做无 pin DBI，bank chaining 用冗余 bank 保持全宽对称 channel，thermal-aware refresh（4ms@105°C）与交错 RS ECC 靠"深 bank 化、每 bank 仅 1364 行"的 bank 几何实现低开销可靠刷新。封装路径：逻辑-DRAM 堆叠经混合间距 C4 bump（min 110µm）到 3D CoWoS interposer，再扇出到 9-4-9 有机基板（MCM 422W TDP、结温 105°C），并有 8 颗 on-package LPDDR5X-9600（每 MCM 128GB 次级内存 tier）承接放不下的权重/KV。热反转优势：逻辑在上、DRAM 在下使 DRAM 比逻辑低 ~3.5°C（与 HBM 的 DRAM 吸收逻辑热量相反）。实测 700MHz 下 2.5ns flit 延迟、105TB/s/card（12.5× HBM3、即使 HBM4 翻倍仍 6.25×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：F2F 键合（36µm-pitch µbump）+ 深 bank 化 DRAM die 组织 + 逻辑 die（N4P）与内存 die 协同设计，bank 支持冗余/每 bank 可配置，映射成 gang/slice 层级 channel；2-High/4-High 堆叠与 DRAM-on-top 封装在实验室测试中（论文以 2×/4× 容量与带宽的缩放点建模其潜力）。使用方式：作为生成式推理（尤其内存受限 decode）的主内存基板——权重与 KV cache 放 3D-DRAM（32GB/card），大模型/长上下文放不下时用 LPDDR5X 次级 tier 或扩展卡数；对比项 XPU+SRAM（150TB/s/4GB）与 XPU+HBM（18TB/s/192GB）。局限：需要围绕 3D-DRAM 重构整个内存子系统（channel 化、映射、可靠性/热），不是即插即用外设。

WaferBRAIN 补充视角（ISCA'26，神经形态突触存储）：WaferBRAIN 把 3D-DRAM（logic-on-DRAM 垂直堆叠基板）用作神经形态系统的突触/连接存储基板：每个计算 die 垂直键合（hybrid bonding）专用 3D-stacked DRAM die（40GB/die，215mm 晶圆共 ~1.92TB），承载全部分突触权重与 fan-in/out 元数据；热神经元状态（膜电位、refractory 标志、每步事件队列）留在片上 SRAM。针对 NAHP 双寻址模式，3D-DRAM 内组织为两类 contiguity-aware 邻接块（紧凑 pointer/length 头 + 连续邻接表）：本地（neuron-driven）布局按 LNid 索引 (L.Syn.Pointer, Fanout) + 连续 <DstNeuron, Weight> 邻接表；全局（axon-driven）布局为 Global Axon-out Index（源 LNid → Axon-out.Pointer/Fanout）→ Global Axon-out 记录 <DstNode, GAid> → Global Axon-in（GAid 索引）→ Global Synapse 连续 <DstNeuron, Weight> 列表；邻接块连续布局配合 coalesced DMA 平衡 compute-communication-memory 并摊销 DRAM 延迟。3D 集成 DRAM 的高聚合并行带宽（面积密度 ~0.43-0.66Gb/mm²）与稀疏 spike 事件访问模式匹配，把每节点突触存储（S_synapse=1.54GB/node：F=256 fanout、16bit 权重+21bit 目标 ID=37bit/synapse）从 SRAM/crossbar 的容量墙中解放。

涉及论文标题：
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
