## Hybrid Bonding（混合键合，3D die 堆叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HB 是 3D 集成技术：两片晶圆/裸片的铜-铜金属互连与氧化物介质直接键合（无凸点、无焊料），键合界面同时提供电气与机械连接。核心指标：互联密度 10K–100K interconnects/mm²（远超 µbump 的百级/mm²）、能量 0.05–0.88pJ/b（比 off-chip HBM 高效 >200×）、速率可达 6.4Gbps 以上；商用主流 pitch 0.8–9µm，亚微米 pitch 已在研究中验证。分 W2W（晶圆对晶圆，适合同尺寸晶圆、良率匹配要求高）与 C2W（芯片对晶圆，适合异尺寸 die，如 AMD 3D V-Cache 把 SRAM cache die 叠在计算 die 上）。TSMC SeDRAM（IEDM 2020）用 W2W HB 把 eDRAM 叠到逻辑上：34GB/s/Gbit、0.88pJ/b。CompAir 用 HB 把 DRAM die 与逻辑 die（SRAM-PIM + NoC）每 bank 1:1 配对堆叠（256 bonds/bank）。CAPA 用法与指标：AMD MI300X 的 SoIC 中 2 个 XCD（N5，~125mm²）经 9µm pitch HB 叠在 IOD（N6，377mm²）上；键合碳 C_bond=(EPA_bond×CI×A_bond)/Y_bond（Eqn. 16），默认良率 HB 95%（TCB 96%）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 的用法：上层 1y-nm 32MB DRAM-PIM bank（≈1mm²）与下层 28nm 逻辑 die（4×8KB SRAM-PIM 宏各 0.136mm² + 4 router + HB I/O）堆叠。面积匹配是硬约束——HB 要求上下层面积相近，所以"为什么不用加速器"：同算力 28nm 脉动阵列综合面积 0.736mm² = 8KB SRAM-PIM 的 5.411×，无法在 bank 面积内配对，SRAM-PIM 是必要选择。带宽设计：DRAM bank 读带宽 32GB/s（256-bit 宽），HB 6.4Gbps × 256 bonds 与之对等；解耦列译码器（32:1→8:1+4:1）提升 bank 读带宽，扩 bonds 占单 bank 面积 20%，在 >10K/mm² 密度下可行。能量视角：HB 单 bit 仅 0.05–0.88pJ，但 cross-die 通信总量大，是 hybrid 相对纯 DRAM-PIM 的能量增量来源（CompAir 的 energy 分解证实）。CAPA 的堆叠碳聚合：C_3D=(C_bottom+C_top+C_bond)/Y_bond（Eqn. 17）——上下 die 各自碳加键合碳再除以键合良率，多 die 堆叠迭代此式；3.5D 集成中把该 C_3D 作为子树代入 2.5D 公式（Eqn. 18）。KGD 前提：D2W HB 与 chip-last TCB 是支持键合前测试（剔除坏 die）的两种流程。DESSCam 的用法（图像传感器堆叠）：顶层像素阵列（PSC 模拟前端，5 µm 像素，沿用 Sony/Samsung 设计 [48,117]）与底层逻辑层（SSPL 稀疏采样逻辑 + PAC 阵列，6.1 µm 像素）经 HB 键合，每像素仅两条键合信号——Vdiff 数据通路与 SCtrl 控制通路；HB 把模拟前端与数字稀疏采样/计数逻辑垂直分层，实现片内低延迟本地事件处理 + 面积效率（346×260 阵列整芯片 3.414 mm²，底层像素面积开销仅 0.6%）。与索尼三晶圆堆叠 CIS+EVS（ISSCC 2023）同属"堆叠图像传感器"路线，区别在于 DESSCam 的底层不是 ISP/NPU 而是像素级稀疏采样逻辑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
工艺：TSMC SoIC（W2W/C2W）、Intel Foveros Direct、三星 X-Cube；商用案例：AMD 3D V-Cache（SRAM 堆叠）、TSMC SeDRAM（DRAM-on-logic）。实现要点：CMP 平坦化、低温退火、对准精度、铜扩散阻挡层。使用方式：SRAM cache 堆叠（V-Cache）、DRAM-on-logic（SeDRAM、CompAir）、CIM 宏堆叠（H2-LLM 等 hybrid-bonding PIM 工作）。选型注意：W2W 要求上下晶圆 die 尺寸匹配（CompAir 的 bank:逻辑面积匹配即此约束），C2W 更灵活但吞吐成本更高。

DIAMoND 的 near-DRAM 用法（ISCA'26）：3D-stacked DRAM（4 层 × 每层 2×2 tiles、共 1.5GB、每 tile 96MB）经 hybrid bonding + mini-TSV 连底部 Logic & Control die（4 个 tile 的 PE 阵列 + SRAM I/O buffer + softmax/SiLU 单元），带宽 1620GB/s、DRAM 面积 48mm²、功耗 3.6W——该形态直接源自 UniIC SeDRAM 多层阵列路线（VLSI 2023 "135 GBps/Gbit 0.66 pJ/bit stacked embedded DRAM by fine-pitch hybrid bonding and mini-TSV"，即 DIAMoND 引文 [63]）。DIAMoND 的分工逻辑：DRAM 侧承担 self-attention 与 KV cache（高耐久、低 P/E 延迟，绕开 NAND 10^3 P/E 耐久限制），逻辑 die PE 阵列做 8-bit VMM（每 PE 十六个 1×16 MAC、2 TOPs@1GHz）。

HybridSpec 补充视角（ISCA'26，HB 作为高带宽-draft 侧内存基板）：HB 的关键取舍是"带宽 vs 容量"——单 DRAM tier 在 reticle 限制下通常 ≤20GB，多层堆叠需要 face-to-back 集成 + mini-TSV 穿透中间 DRAM die，且 mini-TSV pitch 需匹配 HB 的 2-3µm 细 pitch（高深宽比、严格对准容差、散热变差），制造难度大（即便 TSV pitch ~10µm 的 HBM 也花了十多年才到 12 层）。HybridSpec 据此选成熟廉价的 face-to-face 单层堆叠（DRAM tier + logic tier 通过 HB via 直接互连），流片原型 408mm²（24.7×16.47mm）@400MHz：DRAM tier 四颗 2.5GB die（每颗 80 个 I/O group × 256-bit、共 10GB、4TB/s 聚合带宽、DRAM 能量 0.88pJ/bit），logic tier 四个 logic block（各 140KB activation SRAM + 512KB distributed weight SRAM + 80×64 FP16/BF16 MAC），Fig.13 给出 floorplan、HB via 布局与 TEM 照片。与 PIM 对比：HB 在独立工艺分别制造逻辑与 DRAM、再键合，DRAM 容量保留更好、计算单元更强（PIM 在 DRAM 工艺内嵌逻辑通常牺牲 ~50% 容量）；HB 细 pitch 键合 via 只占 DRAM die 面积 <3%。用途：把内存受限的 draft 模型放 HB 栈吃高带宽（4TB/s）而容量只放小 draft + 其 KV cache。

从芯片设计角度拆解（HybridSpec 例）：DRAM die 沿光刻划线从晶圆切出，每 die 多个 bank 独立工作、每 bank 关联一组 HB I/O 与下方 logic die 对齐；logic die 按 DRAM-die 足迹排列 logic block，die 面积决定 bank 数与 HB I/O 位数（带宽与容量随 die 面积缩放）。数据流：draft decode 时权重/KV 从 DRAM tier 经 HB I/O 进 logic block 的 SRAM/计算阵列 → 阵列内 MAC 计算 → 结果经 HB 或块间互连回流；块间用相邻 interconnect 通信（配合 tile 化 TP 的 ring 通信，见 kernel 层 TP 条目）。封装链路：logic die 到 substrate 用 wire bonding（SD 使 XPU-HB 数据移动 <1% 执行时间，不需要 HBM 式 TSV/interposer 高成本连接）。

实现与使用：工艺线 = TSMC SoIC / Samsung X-Cube 的 Cu-Cu 键合（CMP 平坦化 + 低温退火 + 对准）；HybridSpec 用 face-to-face 双层堆叠（成熟路线，2-3µm pitch），多 tier 需 mini-TSV（研究/高成本路线）。使用要点：HB 高带宽适合内存受限算子/模型（draft decode），但容量是硬约束——不能把随请求增长的目标 KV cache 放 HB 栈（HB-ATTEN baseline 因此 TTFT 剧增），这是"带宽-容量分开优化"（HB 供带宽 + LPDDR5X 供容量）的设计依据。

Omelet 补充视角（ISCA'26，2.5D/3D chiplet 分层互连模拟器）：Omelet 把 bonding 技术当作 die-to-die 链路组件链中的可选项之一，用 EM+SPICE 表征四档键合密度（solder ball 30µm、µbump 10µm、Cu-Cu TCB 5µm、hybrid bond 1µm pitch，I/O pitch 覆盖约 1–50µm），给出"系统级带宽并不随键合密度单调获益"的仿真证据：更高密度 bonding 增大 NoI 链路带宽（λ=可用 bonding perimeter/bump pitch → W 更宽），但也放大 NoC 注入宽度与 NoI 链路宽度的失配——低负载下固定打包/去打包（packetization）与 PHY 适配开销占主导，零负载延迟反而更高；深饱和下由拥塞边界端口的 service rate（更宽 NoI 链路排空队列更快）决定，延迟增长略缓（Takeaway 4：更高密度键合提高服务容量但不保证所有负载区间更低延迟）。3D 垂直路径按 F2F/F2B 方向建模（差异在 TSV 于 bonding 界面前/后穿过 die）。Omelet 还强制技术兼容性约束（如 hybrid bonding 仅与 silicon interposer 兼容，因需受控氧化物界面）。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
- DESSCam: An Event-Driven Architecture with In-Sensor Epitopological Sparse Sampling to Break the Latency-Power Tradeoff in Eye Tracking
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture
- Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures
