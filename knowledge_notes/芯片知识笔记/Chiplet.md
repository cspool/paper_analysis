## Chiplet

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chiplet 是把传统单片大 SoC 拆分为多个小裸片、经先进封装（2.5D/3D/3.5D）重组的芯片组织方式。动机是良率经济：小 die 良率显著高于大 die（负二项模型下面积↑→良率↓），多小 die 组装比假想单片省成本最高 40%（Naffziger et al., ISCA 2021），且支持异构工艺节点混合（如 MI300X 的 N5 XCD + N6 IOD）。代价：die-to-die 通信面积开销（D2D PHY）与键合良率惩罚。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CAPA 把 chiplet 拆分度建模为三因素权衡：小 die → 单 die 良率↑（碳↓）vs 组件数↑ → 键合良率浪费↑ vs D2D 面积↑ → 单 die 面积↑（碳↑）。具体试验（MI300X 变体，性能经 D2D 调整保持相当）：2IODs（4→2 个 IOD、每 IOD 4 个 XCD，扣 D2D 面积）整包碳与原设计接近但键合开销更小；8IODs（4→8 个 IOD、每 IOD 1 个 XCD，加 D2D 面积）整包碳更高——小 IC 良率收益被键合浪费淹没。即"拆得越细 ≠ 碳越省"，最优拆分度由键合良率与 D2D 面积共同决定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现生态：UCIe 标准 D2D 接口、CoWoS/EMIB/Foveros 封装载体、SoIC HB 垂直堆叠；MI300X 是 3.5D chiplet 的商用代表（4 SoIC + 8 HBM3）。碳建模使用：CAPA 的 N-ary 树 + 各组件参数即 chiplet 系统描述，arch.json/chiplets.json 输入（附录 Fig. 26-27），用于早期设计阶段的拆分度/集成技术/键合策略权衡；注意 wafer-scale 的 chiplet 集成（多 chiplet 2.5D 于晶圆级 interposer）也在其覆盖内。

DICE 补充视角（ISCA'26）：chiplet 在 AMD EPYC 式组织中是 CCD（Core Complex Die，先进制程计算小片，含核、私有 L1/L2 与共享 LLC）+ IOD（I/O Die，落后制程，聚合内存控制器、DMA 与 I/O）经 PHY 链路互联的结构；CCD-CCD 通信经中心 IOD 中转，使 chiplet 处理器表现为微型 NUMA 系统（AMD EPYC 不维护跨 CCD 一致性、LLC 每 chiplet 私有；Intel Sapphire Rapids / AMD 3D V-Cache 采用全局共享 LLC）。DICE 关注点：随 UCIe 等标准把 D2D 符号率推至 32 GT/s，chiplet 短距链路逼近信号完整性极限（噪声/串扰/损耗↑），PHY 的动态行为（SNR×FEC 码率×迭代解码收敛×重传的耦合环）成为必须模拟的一阶效应；DICE 在 gem5 中运行时建模该 PHY 后，发现固定延迟模型（HeteroGarnet）相对 PHY 保真模型的 IPC 平均偏移 6.8%、最高 27.6%，且其 geomean IPC 与单片基线不可区分——chiplet 架构研究结论本身会被 PHY 建模精度翻转。

Dorado 补充视角（ISCA'26）：chiplet 式簇组织作为 1024 核一致性的载体——32 簇×32 核，每簇一个本地网络（switch 连目录-LLC 分片与核+私有缓存），簇间按 die-to-die 建模（跨簇往返 60 cycles，引用 chiplet 延迟数据 [27]），簇内外均为 2D mesh。行地址哈希决定 home 簇/分片，簇是"本地 vs 远端"一致性事务的天然边界（Temporary home 在簇内消化事务）；扁平协议下扩容只需同构加簇（所有硬件相同），与层级协议需新增不同类别的 cache/目录层硬件形成对比——chiplet 同构模块化是 Dorado 可扩展性的前提。实验对比（Hier2/4/16 vs Dorado）与 PointerSpace 溢出率扫描（64cl_16co / 32cl_32co / 16cl_64co）都按簇/簇大小作为组织参数。

MTIA 300 补充视角（ISCA'26，Meta 首款训练芯片的 chiplet 化）：MTIA 300 采用 compute chiplet（12×6 PE 网格 + 16 ME，25.6×31.4 mm 单 reticle 尺寸 die）+ 2 个网络 chiplet（25.6×9.3 mm，各集成 6 个定制 800 Gbps RDMA NIC）的 chiplet 架构，经 2.5D CoWoS（50.3×51.9 mm 3.2× interposer）与 6 个 HBM3E stack 共封装；die-to-die + 112G SerDes 实现 chiplet 间高带宽密度。网络 chiplet 把 RDMA NIC 直接封装进芯片包，避免 PCIe 数据通路（scale-up/scale-out 网络共用 12 NIC、可灵活切分）；compute die 为 reticle 限制尺寸，故加入冗余 PE 行提升良率（每列容忍 1 个坏 PE、boot 时配置、软件透明）。chiplet 化动机：网络功能模块化复用 + 良率（小网络 die 成本低）+ 独立演进（MTIA 400/450/500 沿用该设计原则）。

Omelet 补充视角（ISCA'26，2.5D/3D chiplet 分层互连模拟器）：Omelet 把 chiplet 系统组织作为模拟器输入（chiplet 数 1/4/8/12、每 chiplet 16 核 4×4、stacking depth 1(2.5D)/2/3、D2D 距离、keepout zone），并针对"同一 chiplet 部署到不同封装平台性能不同"的开放 chiplet 生态问题建模。基线 2×2 chiplet grid 参照 Intel Sapphire Rapids（4 chiplet）、AMD MI250X（2 GPU die）、MI300/EPYC（8–12 chiplet）。关键设计：NoI router 作 indirect router（只转发，不接注入/排出节点）、NoC router 作 direct router（连 core，既是源也是宿）；placement-aware link construction 按 x,y,z 坐标决定 2.5D lateral / 3D vertical 连接；物理约束（beach-front 共享周长、垂直重叠区限制 TSV/hybrid bond 数、interposer span、TSV 密度）参与带宽核算；router-to-router pitch 由 Kite 的 2.2mm 缩到 1mm 以保证 7 种拓扑封装可行（对角/长距连接需动态调整 L/S）。DSE 引擎支持 topology×placement×packaging 联合搜索（穷举 ≤10³ 点、更大用模拟退火），输出 Pareto 最优设计集。

PhaseWeave 补充视角（ISCA'26，数据中心异构 chiplet 服务器）：chiplet 不只为良率/模块化，还可做"每 chiplet 专精一类工作负载 phase"的异构组织——PhaseWeave 服务器集成 4 类 chiplet（high-compute 高频率宽发射大 ROB/SIMD、fast-memory 大 L2+高带宽 DRAM 通道、near-network die 内集成 NIC、low-power 精简低功耗核），全部同 x86 ISA（iso-ISA）实现线程透明迁移；chiplet 内核心按 2D mesh 组织（3 cycles/hop），chiplet 间 high-bandwidth all-to-all 互联（60 cycles、1Gbps/link）；内存物理异构：统一地址空间但每 chiplet 挂专属 DRAM 分区（memory channel/DIMM 粒度，带宽延迟按 phase 需求配置，Mem BW 17.06-25.60GB/s、Lat 15-22 cycles），配 NUMA 式页迁移与两级 MESI 目录一致性（global home + 每 chiplet local home，intra/inter-chiplet directory）。chiplet 化收益：per-core 面积降 27%/11%/34%/39%（compute/fast-mem/near-net/low-power）、功耗降 34%/33%/35%/56%，使 iso-area 下容纳 38 核（vs 同构 28 核）——"每核专精一类"取代"每核兼顾全部"是 chiplet 异构的核心理由。

CASCADE 补充视角（ISCA'26，TFHE 加速器的多芯粒流水线化）：CASCADE 用 chiplet 化解决"跨 HMUX 流水线并行的内存带宽"问题——12 个 HMUX Chiplet（HC）按 4×3 网格、环形拓扑经 UCIe D2D 链路互连，把集中式大容量内存转换为分布在每个 chiplet 的小型 SRAM（每 HC 10.5 MB BSK SRAM + 1 MB 缓冲，共 126 MB BSK 驻留）。chiplet 化动机：(1) 大规模分布式 SRAM 的集中式单片实现成本/良率不可接受，拆成小 die 分布在成熟高良率区（单 HC die 面积 50-150 mm²，DSE 最优 C=12，C>12 时每芯粒 D2D PHY 面积税超过并行收益）；(2) 分布式内存把高并发 BSK 访问限制在各 chiplet 本地（对比集中式 HBM：MP-PP 使能流水后利用率仅 14.3%，MP-PP-HBM 需 8 个 HBM stack 且每瓦效率低 3.7×）。封装采用成熟的 2.5D 无源硅中介层（避免激进封装，整体规模/互连密度/封装复杂度在 Intel Sapphire Rapids 等工业多芯粒产品已验证范围内），与 MTIA 300 的 2.5D CoWoS、PhaseWeave 的异构 chiplet 同属"chiplet 作为可扩展/专精化载体"路线，但 CASCADE 的 chiplet 是流水线阶段（数据依赖链）而非任务并行单元。

涉及论文标题：
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
- DICE: Detailed Inter-Chiplet End-to-End PHY Modeling for Accurate Chiplet Simulation
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
