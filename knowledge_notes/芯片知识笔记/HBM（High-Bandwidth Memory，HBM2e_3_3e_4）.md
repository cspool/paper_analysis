## HBM（High-Bandwidth Memory，HBM2e/3/3e/4）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HBM 是 JEDEC 标准化的高带宽内存：多个 DRAM die 经 TSV 垂直堆叠在 base die 上形成堆（stack），再经 μbump 装在硅 interposer 上，通过超宽接口（HBM3 1024-bit、HBM4 2048-bit）提供 TB/s 级带宽，是 AI 训练 GPU 的主存。各代容量/堆高：HBM2e 8Hi/16GB、HBM3 8Hi/16GB、HBM3e 12Hi/24GB、HBM4 16Hi/48GB（JESD270-4，2025-04 发布，~2TB/s/stack、32 通道）。CAPA 用 TechInsights 的碳数据（Table III）：每堆 kgCO2eq 18.16/19.95/27.83/43.50，折合每 GB 1.135/1.247/1.160/0.9063。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CAPA 模型 C_HBM = C_per_GB × Capacity（Eqn. 13）——因 JEDEC 标准化使硬件组成可预测，per-GB 碳是合理指标。在整包碳中 HBM 是最大单项：A100 的 6 堆 HBM2e 共 108.96 kgCO2eq，占整包 47–52%；MI300X 的 8 堆 HBM3 同样主导（Fig. 16a）。HBM 还通过 PHY 面积参与 interposer 建模：HBM PHY 面积按 JEDEC 各代规格计算，计入 interposer 的 D2D 金属面积 A_D2D,total。数据不确定性处理：NVIDIA HGX H100 PCF 报告反推 per-GB 仅 0.71–0.85 kgCO2eq 且未说明是否计入第 6 堆未激活 HBM3，与 TechInsights 的 1.247 差距大，CAPA 因此在合适处给出区间（Fig. 17：HBM 低碳假设下 MI300X 整包 −27%）。设计启示：HBM 碳占 ~50% 且敏感性高（H100 碳/GB 扫 0.7–1.3 对整包影响大），架构师要审慎选择 HBM vs DDR/GDDR 与容量，软件要充分利用以摊薄。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
制造上：TSV 堆叠 + base die + interposer 装配（CoWoS-S 是主要载体）；业界 2026 年 HBM4 量产（三星 12 层 36GB 首发、SK hynix 16 层），JEDEC 放宽封装高度至 ~775µm 使 16 层可用 μbump 实现、推迟了 HBM 的混合键合化（预计 HBM4E/HBM5），另有免 interposer 的 SPHBM4（JESD330-4）变体。碳建模上的使用：per-GB 查表 × 用户输入的容量；对 H100 这类"6 堆只激活 5 堆"的设计，PCF 反推与 TechInsights 的区间即反映了"未激活堆是否计入"的账务歧义——这是碳核算而非性能层面的歧义。

CODO 的 HBM 使用视角（ISCA'26）：AMD Alveo U280 的 8GB HBM 作为 FPGA 数据流加速器的片外存储，编译器自动管理其访问模式——codo-transmit 命令自动生成 host code 与 burst 访问操作，把模型权重等参数分布到多个 HBM channel 并行访问独立内存区域、提升带宽利用率（用户可指定 channel 数）。效果：GPT-2 上板执行时间分解（Fig. 12）显示数据搬运占比低（短序列略高、长序列计算主导，自注意力计算增长快于数据搬移）；GPT-2 对比实验中 CODO 跑 8GB HBM 的 U280，反超 16GB HBM 的 U55C 上的 StreamTensor 1.23×，说明 HBM 带宽利用率（burst + 多 channel）比容量更关键。

Raptor 补充视角（ISCA'26，带宽扩展与 I/O 功耗的对比基线）：HBM 靠加宽接口与提速扩展带宽——HBM3 保留 1024-bit 数据接口但把通道翻倍为 16×64-bit（提升并发）；HBM3E 把 pin 速率推到 >9.2 Gb/s、单 placement 超 1.2 TB/s；HBM4 接口翻倍到 2048-bit、通道 32，单堆达 2 TB/s，但接口复杂度与 I/O 功耗随之上升（JEDEC JESD238A）。Raptor 以此为对照论证 3D-DRAM 的优势：decode 是字节搬运主导（每 token 读全部累积 KV），传输能量 pJ/bit 直接决定 tokens/J——HBM 的每 bit 能量不随代际有效下降；HBM 的 burst 多周期传输 + PHY 级 DBI（每 beat 决策、专用 DBI pin）在 3D-DRAM 的单周期宽 µbump 接口上不可用，而 stream flipping 在架构层实现等效（-18% I/O 能量，0.45→0.376 pJ/bit，比 HBM3 低 ~6×）。实测 3D-DRAM 700MHz 下 105TB/s/card = 12.5× HBM3 卡，即使 HBM4 带宽翻倍仍有 6.25× 优势；XPU+HBM（18TB/s/192GB）作同逻辑 baseline 时 tok/s/card 比 3D-DRAM（100TB/s/32GB）低 4.71×、TPOT 高 9.96×。

FlashTFHE 补充视角（ISCA'26，两栈 HBM2E 作 TFHE 加速器片外内存）：FlashTFHE 以两栈 HBM2E（819GB/s）为唯一片外内存，承载数 GB 的 BSK/KSK（multi-bit 参数下 GPT-2 decoder layer 的 key 达 4.7GB，远超片内 45MB SRAM）与 GLWE/LWE 密文流。带宽分析（Figure 15a）：BSK/KSK 带宽因片内 chunk 复用 + 核间共享而恒定，GLWE/LWE 带宽随核数 2→8 线性增长，两栈 HBM2E 在 8 core 内不构成瓶颈；对比实验用双 AMD EPYC 9654（921.6GB/s，超过两栈 HBM2E）证明提速主要来自微架构而非带宽。论文未开源 DRAMSim3 配置（联网未找到仓库）。

GenZA 补充视角（ISCA'26，ZKP 加速器的 HBM2e 片外内存）：GenZA 配备 2 个 HBM2e 接口共 1 TB/s off-chip 带宽（References [33] HBM2E 白皮书、[55] Fine-Grained DRAM），承载 NTT/MSM/sumcheck/多项式 kernel 的中间数据与 EC 点/标量流（如 2^23 NTT 单次流量 7.4 GB）。设计权衡：用 HBM 带宽换取片上 SRAM 容量——MSM 桶数超片上 SRAM 时多轮从片外重载数据（每轮加载全部数据），动态 window size 成本模型显式把带宽约束作为选择 c 的输入；NTT 折叠流水线把 off-chip 流量减半（2^23：7.4→3.0 GB）。HBM PHY 是面积功耗大头（2 个 PHY 29.8 mm²/31.7 W，占整芯片 58.5 mm²/64.1 W 的约一半）。可扩展性实验（Table XII）显示带宽从 0.5×→4× 对性能的边际收益递减（Groth16 97.8→52.4 ms），表明其 kernel 混合下带宽压力中等、PE 数（计算）是更强杠杆。

HBM-CASO 补充视角（ISCA'26，HBM 物理组织与 ECC 重组的芯片设计约束）：HBM3 每 core die 含 4 channels、四高堆共 16 channels，每 channel 分为两个共享命令总线的 pseudo-channel；pseudo-channel 访问粒度 38B = 32B data + 2B metadata（CRC）+ 4B parity，经 38 个 TSV I/O 以 8-bit burst 传输（32 数据 TSV + 校验/元数据 TSV）。64B cacheline 由两个 pseudo-channel 串联（channel 粒度）访问。HBM-CASO 的芯片设计要点：(1) 为给 advanced SysECC 腾出 8B parity 空间中的 4B，把每个 64B 写的 4B local ODECC parity 经 Merging Unit 线性合并成 2B regional parity（两个 16B 数据块上的 RS(18,16) 合并为 32B 区域上的 RS(34,32)），且合并不跨 pseudo-channel（p_local_0/p_local_1 来自同一 32B 区域）——避免跨 PC 的物理对齐问题；(2) 把原 4-bit CRC 通道改送 global SysECC parity（原 4-bit 只做检错、无纠错，改后传递 4B 系统纠错码）；(3) 延迟写验证的两个 Accumulation Unit 均匀分布在两个 pseudo-channel 上，各自 XOR 累加 2B regional + 2B CRC parity（每 PC 一个、不跨 PC），最终 64-bit XOR 结果在批末与控制器比较；(4) mode register 新增 HBM-CASO 模式位（可全局静态或按块动态启用，块级用 metadata 内 flag 指示），与 power-down/self-refresh 等既有 DRAM 模式并列。芯片代价（45nm Synopsys DC 综合）：OD 侧仅 +61 cells/+17μm²（Merging Unit + 两个 Accumulation Unit），片上纠正延迟几乎不变；较重的 global/regional SysECC 编解码器移到控制器侧（R-mode 2100μm²/2.52ns 纠正、G-mode 5910μm²/4.56ns 纠正），并把额外延迟 0.25/0.51ns 建模进 tCL。

HE² 补充视角（ISCA'26，HBM 作 NMP 计算基板 xMU 的芯片设计约束）：HE² 把近存计算单元（xMU）直接集成进两个 HBM2 栈（合计 8 GB、1 TB/s、JESD235）的 column decoder——每 PE 从 global row buffer 取 256-bit 进 local buffer 隐藏 bank 访问延迟，row-major 数据布局把每个密文多项式摊到所有 bank 使每 PE 本地取操作数。芯片设计约束与权衡：(1) 只放轻量 SIMD MemOps（CtAdd/PtMul/IP/Autom），避免在 DRAM 内集成复杂 ComOps 逻辑（DRAM 工艺晶体管慢 ~3×、逻辑密度低 ~10×，面积预算受限，遵循 AiM [23]/Newton [13] 类 bank-level 先例）；(2) xMU PE 用 12nm PDK 综合（与 HBM 逻辑兼容），占 HBM 模块面积仅 11.1%，峰值功耗控制在 all-bank-interleave 访问预算内、运行于 85°C 热包络内，RTL 验证时序且保持 HBM bank I/O 兼容；(3) MemOp fusion 消除顺序 MemOps（IP/PMul）间的 row-switch 写回开销；(4) in-DRAM automorphism 复用原生 DRAM 数据通路（global row buffer 2048 coeff/cycle 站内、bank I/O controller 128 coeff/cycle 站间、GBus 32 coeff/cycle 组间，借鉴 Figaro [51]）实现零新增逻辑的密文旋转，比 F1 [43] 两级 automorphism 快 1.10×。带宽/容量权衡：8 GB 容量足以容纳 HERO 最优 PKB 融合的 evk 工作集；敏感性显示 0.5 TB/s 带宽即可达 SHARP 同级性能（Fig. 17），说明通信优化（HERO + 双级流水）显著降低了对 HBM 带宽的需求。

HybridSpec 补充视角（ISCA'26，HBM 作为"高带宽-低容量-高成本"的对比基线）：A100 采用 80GB HBM2e + 硅 interposer 封装，HBM 估占 A100 成本（~$8,000）一半以上（TSV + interposer 装配成本高）。HybridSpec 用它做 XPU 内存替换对比（Fig.19）：40GB/1555GB/s HBM2 与 80GB/1935GB/s HBM2e 在低请求率（1 req/s）下因高带宽对 LPDDR5X（1.1TB/s）有显著加速，但请求率上升后 KV cache 增长使容量成为瓶颈（40GB 配置甚至劣于 LPDDR），批量受限、新请求等待内存释放、延迟上升——实证"容量不足在在线 serving 中会抵消带宽优势"。成本视角：HBM 占 A100 成本过半，HybridSpec 用 LPDDR5X（~$400/包 ×8）+ HB 栈（~$400/400mm²）替代，总内存+封装约 $3,600，成本更优。TSV pitch ~10µm 是 HBM 相对 HB（2-3µm）带宽密度受限的结构性原因。

从芯片设计角度拆解（对比例）：A100 的 HBM2e 经 TSV 堆叠 + base die + interposer（2.5D）装配，多堆并联提供高带宽但总量固定（80GB），LLM serving 中 KV cache 增长直接封顶可并发请求数；HybridSpec 用"HB 栈（4TB/s，放 draft）+ LPDDR5X（512GB，放 target 权重与 KV）"把带宽与容量分到两个物理内存基板，避免 HBM 单基板的双重约束。

实现与使用：HBM 是 GPU/TPU 标准主存（JEDEC 标准化、TSV 堆叠 + CoWoS interposer）；HybridSpec 的用法是把它作为对比 baseline 而非组件——论证"高带宽但容量有限/成本高"的基板不匹配 LLM 在线 serving 的容量需求，而高带宽需求可由专用 HB 栈承担（draft 侧）、容量需求由 LPDDR5X 承担（target 侧）。
RangeGuard 补充视角（ISCA'26，HBM 现场可靠性缺口与 ECC 预算约束）：RangeGuard 的动机数据——HBM 因 2.5D 集成（TSV/μbump、晶圆减薄、热循环）与更高工作温度比 DDR 脆弱：ByteDance 10 万服务器 DDR4 平均 0.07 errors/device/month [3]，而 Huawei 1.5 万加速器 HBM2 core die 平均 35 errors/device/month [4]（约 500× 差距）；Meta 405B 训练 54 天在 16,384 个 H100 上报告 72 次不可纠 HBM3 错误（S-ECC 为 SEC，每 ~18 小时一次）[23]。物理根源：HBM3 sub-wordline（SWL）driver 每访问扇出 32 个数据 bit、以高电压运行，单驱动故障可一次破坏 32-bit burst，超出 HBM3 O-ECC 16-bit 符号纠正能力并暴露到主机。GPU 内存的芯片设计约束：每 256-bit block 只留 2B（16 bit）parity（6.25%，对应 HBM 32B 访问粒度），RangeGuard 的目标就是在该固定预算下用 RID 语义保护容忍 64+ 个翻转 bit。芯片组织上，HBM3 每 core die 4 channels、四高堆 16 channels，每 channel 分两个共享命令总线的 pseudo-channel（HBM-CASO 条目的 38B 粒度即 32B data + 2B metadata + 4B parity）。
TDMSim 补充视角（ISCA'26，GPU DRAM cache 全系统模拟的主存配置）：TDMSim 在 gem5 中构建 AMD MI300X 单 XCD CPU-GPU 系统，主存为 2 个 HBM3 stack、共 24 GiB、660GB/s 总带宽，FR-FCFS 调度、64 项读队列/128 项写队列；HBM 时序参数（Table II）：tRCD=12、tRCD_WR=6、tCCD_L=2、tRP=14、tRAD=28、tCL=18、tCWL=7、tRRD=2、tXAW=16、tRL_core=2、tRTW_int=1。角色定位：HBM 作为 LLC（32MiB SRAM 或 2D DRAM cache）之下的片外主存，承接 LLC miss；把 HBM stack 数与对应 cache 容量按比例缩放以保持系统代表性。研究焦点是 LLC 配置（SRAM vs 硅 1T1C vs 2D 1T1C/3T0C）对 GPU 性能/能耗的影响，HBM 本身为对照基板（FR-FCFS 时序由模拟器建模）。

XtraMAC 补充视角（ISCA'26，Alveo U55c 的 HBM channel 组织作 GEMV 带宽基板）：AMD Alveo U55c（xcu55c-fsvh2892-2L-e）集成 32 个 HBM channel、总带宽 460 GB/s，HBM 控制器位于 Alveo static shell、不计入用户逻辑资源。XtraMAC 的 tile 并行 GEMV kernel 按 channel 组织：每 HBM channel 提供 512-bit 接口，每 cycle 读出 512-bit 权重字拆成 per-lane 权重段分发到该 channel 对应 PE 内级联的 XtraMAC 链；每 channel 级联数 N_MAC=BitWidth_channel/(BitWidth_weight×P)，INT4 权重 + P=2 lanes 时单 channel 64 个 MAC 输入/cycle；32 channel 理论 2048 个实例、实际 1920 个（30 活跃 channel 算 GEMV、留 1 个读激活 + 1 个写回），512-bit 接口配合 2× lane 打包恰好每 cycle 喂满 64 个 MAC。带宽受限特性：GEMV 吞吐上限 ≈ HBM 带宽/权重字节，FPGA 维持 ~74% 有效 HBM 利用率、接近带宽 roofline，从而在 460 GB/s（vs H100 2 TB/s）下仍以 1.2× 时延、1.9× 能量效率反超 GPU。1920 实例时因 HBM 接口附近路由拥塞 Fmax 从 300 MHz 降到 250–270 MHz，但带宽受限下对吞吐影响可忽略。
涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA

HiT 补充视角（ISCA'26，HBM 作为稀疏加速器片外主存的带宽/能量假设）：HiT 以 2 TB/s HBM 作为唯一片外内存（代表当前代 GPU/TPU 的带宽水平），HBM 访问能耗取 3.9 pJ/b（来自 Fine-Grained DRAM 研究 [52]），能量核算 = 计算能量 + HBM 访问能量。芯片设计要点：(1) 片上 Global Memory 为 16MB 多 bank SRAM（512 bank、64B datawidth），tile 数据经 HBM 预载到片上后再流式进 Compute Row 的专属 bank，HS workload 属于 memory-bound——HiT 用 HSparse 的 B 行顺序流式 + 片上 psum 累积来削减 off-chip 流量（与 Trapezoid 的 memory-efficient Gustavson 相当，ca-0.4/ca-0.2 上整个 A 可常驻片上）；(2) 面积/功耗按 22nm 评估（Global Memory 用 CACTI 22nm 估算、HBM 能耗 3.9pJ/b 固定假设），HS 下 9 个 SuiteSparse workload 的 off-chip 流量分解（图 19）用于说明"外积数据流虽然 psum 多，但靠片上累积仍能达到 Gustavson 级的内存效率"；(3) 与 Trapezoid 共享同一 2TB/s HBM 假设、以公平对比——HBM 带宽充足时瓶颈转移到片上存储带宽（register file 4R4W）与计算并行度。


LÆGIS 补充视角（ISCA'26，HBM 作为可信安全基板承载 IV Bank）：LÆGIS 首次利用"GPU 3D-stacked HBM 可信"威胁模型（与 Graviton/NVIDIA CC 一致：HBM 集成在包内、经 TSV 互联，无公开 RowHammer 攻击）在 HBM 内显式存放加密元数据——每个 GPU HBM stack 预留 8 MB（512K 行 × 128-bit）作 IV Bank，按 VABlock（2 MB）粒度存 IV Bank Entry（19-bit ID + 77-bit RV + V/D/O/R 控制位 + 9-bit CTR）。物理组织（Table II-B：8 GB 4-Hi stack、8 channels、16 pseudo channels、1 KB row、FR-FCFS、16 banks、4 bank groups/channel、128-bit 接口 BL4、8 channel @1GHz 共 256 GB/s、CL:RCD:RAS:WR:RP=14:14:33:16:14）：IV 交错分布到所有 channel/bank，1 KB row 存 64 个 IV、每 bank 64 行落在单 sub-array，高地址位（物理地址 bit 17-22）作 row index，每次访问从 16 列聚合一个 entry——均衡负载避免热点，IV 与数据可共享 sub-array；多 stack 扩展只需固件更新物理地址映射（19-bit ID 支持至 1 TB HBM）。芯片设计收益：以 8 MB HBM 元数据换取免除完整性树（对比 SGX 25% enclave 内存元数据开销），且 per-page 粒度使 IV 访问局部性高、片上 16 项 IV cache（256B SRAM）即可覆盖。

MTIA 300 补充视角（ISCA'26，DLRM 训练芯片的 HBM3E）：MTIA 300 采用 6 个十二高 HBM3E stack（compute chiplet 东西侧各 3 个）共 216 GB、6.1 TB/s（R 或 W），配 192 MB 片上 SRAM（LLC/LLS，11.4 TB/s）。设计核心是 HBM bytes-to-FLOPS 比 >2× H100：DLRM 训练 FLOPS 需求中等（~3 GFLOPs/sample）但稀疏特征极大（150B 参数 99% 在稀疏侧、embedding 表常超单卡容量），高 HBM 容量/带宽支撑更大 local batch（10240、24 卡 Perf/TCO 1.42×）与更高 MFU；实测 BF16-add kernel 达 5.57 TB/s（91% 峰值），LLM 推理（DeepSeek-R1、8 卡）高并发下靠 HBM 带宽优于 H200（141 GB/4.8 TB/s）。

ParetoES 补充视角（ISCA'26，FPGA 检索加速器的 HBM2 伪通道使用）：ParetoES 部署于 Xilinx Alveo U280（两栈 HBM2 共 8GB、32 个 pseudo-channel、每 PC 256MB、512-bit 接口、460GB/s 理论峰值），把"每 ACPE 绑定一个 HBM pseudo-channel"作为架构级隔离原语：32 个 ACPE 各持一条专属 PC，形成完全隔离的 compute-memory 对（无跨核通信/同步），使 32 通道在稀疏矩阵流式拉取时持续满载（对比 FPGA32/AccelES 同平台）。芯片设计要点：(1) 每 PC 512-bit @225MHz（U280 官方为 256-bit AXI @450MHz 每 PC，论文按数据通路 512-bit 计），质心/簇子矩阵按 Ultra-CSR 编码对齐 PC 宽度（30 非零/512-bit packet）实现 1 packet/cycle 流式；(2) 簇感知数据布局把随机访问限制在活跃簇块内、呈 burst 流式，配合 URAM 复制（15 路）把 query 读取全片上化，避免伪通道随机读冲突；(3) 2KB Top-512 结果经 PCIe 回传 host，微秒级延迟。设计权衡：带宽 460GB/s 是检索吞吐的硬上界，选择性计算（nprobe 调 Recall）把实际访问量降到全计算的 32–96%（Recall 0.8/0.9/1.0 对应消除 68%/44%/4% 访存）。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
