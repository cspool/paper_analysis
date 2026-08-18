## Wafer-Scale Integration（晶圆级集成 / WSE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wafer-scale integration 指以整片晶圆为单位构建计算系统：一种形态是 wafer-scale processor（如 Cerebras WSE-3，单个巨型 die 布满整晶圆，靠缺陷容忍架构解决良率问题），另一种是"晶圆级 chiplet 集成"（大量 chiplet 2.5D 集成在晶圆大小的 interposer 上，如 Tesla Dojo、多 chiplet waferscale 研究）。两者的共同约束：制造碳与总面积成正比，而良率随面积指数恶化。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CAPA 的覆盖度：晶圆级 chiplet 集成本质是"多 chiplet 2.5D 集成在晶圆级 interposer 上"，现有 Eqn. 15/18 直接适用；但 Cerebras WSE-3 这类单片 wafer-scale 处理器需要专门的良率模型——其"100× 缺陷容忍"架构（按核粒度绕过缺陷、无 binning 概念）无法用 Eqn. 1/9 的整 die 良率描述，CAPA 明确将其列为需要独立 yield model 的扩展点。碳视角：整晶圆的 CPW 全部由功能系统承担（几乎无废片分摊），键合良率与冗余策略决定有效碳效率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：WSE-3（Cerebras）用 1×1 网格冗余路由绕过缺陷核；Dojo 用 25 个训练 tile 的 chiplet 集成；学术原型如 2048-chiplet 14336-core waferscale 设计（DAC 2021）。碳建模使用：CAPA 用户可把 wafer 级 interposer + 大量 chiplet 描述为树直接估算，但对单片 WSE 需扩展新良率模型（论文 Limitations/Extensibility 节）。

ConBin 补充视角（ISCA'26）：把"单片 WSE"路线推演到商用场景——Cerebras WSE-3（~46,225mm²、5nm、缺陷密度 ~0.001 defects/mm²、~970,000 核 mesh）按负二项良率模型产出无缺陷芯片概率实际为零；Cerebras 专利 [39] 的拓扑保持冗余修复（每核 6 条 2-hop 冗余链 + 抽取缺陷无关的子 mesh 作为可用阵列）虽保住功能良率，但激活核数芯片间差异大（128×136 规模 512 实例下平均仅 5122 核 = 朴素"全健康核修复"的 38%、方差近 3×），且通信性能由缺陷分布支配而非仅由核数决定，传统按频率/核数的 binning 失效。ConBin 针对单片 WSE 商用化提出：Performance Binning（按实测负载性能分档）+ 故障相关感知冗余互连设计 + 制造后修复 + bin 感知映射/调度，收敛芯片间性能分布，把同一功能良率转化为更高 premium-bin yield（2.80×）与总可售有效算力 SECC（2.64×）。可扩展性数据：40×48/64×72/96×104/128×136 四规模下冗余设计 1.31–6.28 min、修复 1.17–7.12 s，mesh 相似度 F_norm 保持 86.43%–88.19%。与 CAPA"按核粒度绕过缺陷、无 binning 概念"的描述互补：ConBin 在绕过缺陷之外显式建模"绕过后的性能分布"并为之优化。

BusyBarn 补充视角（ISCA'26，wafer-scale chiplet 集成路线的部署优化）：聚焦"晶圆级 chiplet 集成"这一形态——通过先进封装把密集排列的 known-good compute/memory die 互连成 wafer 级处理器，目标系统为层次化两级 2D mesh（上层 die、下层 die 内 core），die 间经 D2D 链路（多 SerDes lanes）跨硅 interposer 通信，每 die 配本地 DDR/HBM（对比 Cerebras WSE 纯片内 SRAM 空间并行，BusyBarn 系统扩展计算与内存容量以减少服务单一大模型所需设备数）。良率/可靠性视角：论文引用 H100 与 Cerebras WSE-3 缺陷率均约 0.001 defects/mm² [35]，因 wafer 规模大，制造缺陷与运行期退化（电迁移、热载流子注入）必然发生——BusyBarn 采用通信路径级容错（路由 LUT 重配置绕过故障节点/链路，类比 Google TPUv4 流量重路由）而非 Cerebras 式备用 core/互连冗余，以最小额外制造开销使缺陷芯片逼近标称性能；缺陷率敏感实验在 20×20 core mesh 上注入 10%/15%/20% 随机/聚集故障，BusyBarn 相对 Gemini 1.24–1.53× 加速。与 ConBin 的"性能分布 binning"互补：BusyBarn 不做制造后 binning，而是让映射/路由算法在任意故障拓扑上自适应重优化。

MoE serving 补充视角（ISCA'26，Patterns behind Chaos）：把 wafer-scale 路线落到"晶圆级多 chiplet GPU"做 MoE LLM serving 的架构评估——单 die 尺寸受光掩模限制（800-1000 mm²），先进封装（TSMC CoWoS、Samsung X-Cube、Intel EMIB）推动多 chiplet，方向演进出 TSMC System-on-Wafer（SoW，单晶圆容纳最多 24 compute die + 96 HBM die、超 200,000 mm²）。论文评估两种拓扑：Tesla Dojo（5×5=25 die 2D mesh，已部署系统）与 TSMC SoW（8×3=24 die 2D mesh，近未来路线图）。每个 die H100-like：1000 TFLOPS FP16、80GB HBM、3.35TB/s 本地 HBM 带宽、1.7TB/s 相邻 D2D；Dojo-Enhanced 扩展（B300-like：4500 TFLOPS、180GB、8TB/s DRAM、2TB/s D2D）体现未来性能趋势。SoW 的芯片级互联：GPU die 经 local-silicon interconnect（LSI）直连本地 HBM die，相邻 GPU die 经 LSI 垂直或 XSR SerDes 水平互连——LSI 与 SerDes 都提供 terabit 级带宽，但跨 die 访问需多跳、远程 HBM 同时被多个 die 访问产生带宽争用/拥塞，成为主瓶颈。编程模型分两派：multi-GPU-like（WSC-LLM、MoEntwine，暴露 2D mesh、细粒度 die 控制，但 Blackwell/Rubin 不暴露 die 级工具链、MIG 会禁用高速 D2D，近期限难）与 single-GPU-like（HDPAT、Hecton、本文：整 wafer 当单一 GPU，软件不感知跨 die 数据移动，优化负担全部交给硬件）。结论：单芯片装下整个 MoE（>3TB HBM、PFLOPS 级算力、batch>10000），但"本地 vs 远程访问成本差最高 15×"且软件无法控制，架构级优化（放置感知任务分配 + 硬件管理 HBM）成为关键（见硬件架构层两个条目）。

WaferBRAIN 补充视角（ISCA'26，神经形态 3D-WSI）：把 wafer-scale integration 推进到"3D 晶圆级集成（3D-WSI）"形态——215mm×215mm 晶圆上 6×8=48 个计算 die 以 2D mesh 组织，每个 die 经 hybrid bonding 垂直键合专用 3D-stacked DRAM die（40GB/die、共 ~1.92TB/wafer，支撑 1B 神经元/256B 突触），DRAM die 经 TSV 连到底层 RDL、再经 package substrate 键合 PCB（消除长 PCB 走线，D2D 时延 ~1ns 级、带宽 >1Tbps）。与 Cerebras WSE 的"单片巨型 die + 片内 SRAM"不同，WaferBRAIN 采用 chiplet 化 WSI（每 die 23mm×32mm、4×4=16 个 BPU 节点、每 die 含 NoC 路由器与 axon-in/dendrite/soma/axon-out 模块）并依赖 3D-DRAM 提供突触存储。原型 Lyra X（UMC 40nm、12 英寸 228×211mm、11×16 die mesh、SRAM-only 支持 202M 神经元/2B 突触）实测 intra-die 1ns / inter-die 8ns / inter-wafer 493ns 跳时延，作为架构评估的 hop-latency 标定。多晶圆 scale-out 用 switchless dragonfly（见芯片设计库 Switchless Dragonfly 条目）；WSI 相对 PCB 集成（D2D 100ns/100Gbps，Loihi 参数）把可持续 firing rate 提升 13×（PCB 连 0.5% firing rate 都撑不住）。

涉及论文标题：
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
- WaferBRAIN: Whole-Brain Scale Neuromorphic Architecture Based on Wafer-Scale Integration
