# L6: 芯片设计 — 水平分类总结

## 问题覆盖概览

| Q-ID | 覆盖方法数 | 关键方法 |
|------|-----------|----------|
| Q6.1 | 7 | Chiplet (MCM-GPU, DFBM, Active Interposer), Wafer-Scale (WSE-2, SoW/Dojo, Two-Level CP, ATU/PDU, Multi-Die Task Allocation), PIM (DRAM-PIM, 3D NMP), NoC (XY Routing, Multi-VN/VC) |
| Q6.2 | 8 | Chiplet vs Wafer-Scale 全维度对比 (MCM-GPU, WSE-2, SoW, Two-Level CP, ATU/PDU, Multi-Die Task Allocation, DFBM, 3D NMP), 设计空间权衡 (BW/功耗/面积/良率/可扩展性) |
| Q6.3 | 8 | PIM 三大范式 (DIMM级/HBM-PIM/3D NMP), Bank-Level Parallelism, Duplex xPU+PIM, ERAS Expert驻留感知, ReRAM Crossbar, KV Cache PIM卸载, Stratum Mono3D, STARC |
| Q6.4 | 9 | 2D Mesh NoC + XY Routing, Torus HalfRing+DimRotation, Meta MTIA 2i Custom Non-Blocking NoC, DFBM Interposer NoC, ICCA All-to-All vs Mesh 拓扑对比, Mugi 3-Channel NoC, FlashFuser Inter-Core NoC, SCAR MCM Scheduling, FEATHER BIRRD |
| Q6.5 | 11 | Timeloop, MAESTRO, Mess Simulator, gem5+Garnet, SCALEsim-v2, BookSim, DRAMsim3, 自研 Event-driven Simulator, ASTRA-sim, Synopsys/Cadence EDA 工具链, Verilator |
| Q6.6 | 9 | FPGA 原型 (Xilinx VCU118/Alveo/ZCU104), ASIC/RTL (SystemVerilog+Synopsys DC+Catapult HLS), MLPerf, PARSEC, MoE-CAP, WikiText-2, Cerebras CS-2/CS-3, NVIDIA H100/B300-like, AMD MI300X |

---

## 按实验环境分类

| 分类 | 方法 | 具体方法描述 | 硬件平台 | Benchmark | 实现框架 | 来源 |
|------|------|-------------|----------|-----------|----------|------|
| **GPU/NVIDIA** | Multi-Chiplet GPU (MCM) | 多 chiplet 通过 2.5D interposer (CoWoS/EMIB) 集成，每 chiplet 含 64 SM + L1.5 2MB + LLC slice + HBM 80GB；chiplet 间 D2D 768 GB/s (crossbar) 或 1.7 TB/s (mesh)；NUMA 效应 remote/local 延迟比 10-15×；L1.5 cache 专门缓存 remote data 减少跨 chiplet 访问；Single-GPU-like 编程模型 | NVIDIA Blackwell B200 (2-die NV-HBI 10 TB/s), AMD MI300X (8 compute chiplets), Rubin (规划 4 chiplets) | Custom MoE traces, MLPerf | CUDA, vLLM, TensorRT-LLM | Q6.1, Q6.2, Q6.6 |
| **GPU/NVIDIA** | Wafer-Scale Multi-Chiplet GPU (Orders in Chaos) | Two-Level CP (Global CP A76-class 1.1mm² + 25× Local CP A72-class 0.3mm² each) 实现 expert-placement-aware 任务分配；Multi-Die Task Allocation (Alg.1: Candidate Mechanism + Block-Granularity block=50 + 三维 Cost Model)；ATU (4.25KB SRAM) + PDU (128B register file) hardware-managed HBM caching；总面积/功耗 overhead <0.04% | Tesla Dojo 5×5 (25 dies, 25K TFLOPS FP16, 2TB HBM), TSMC SoW 8×3 (24 dies + 96 HBM dies, >200,000mm²), Dojo-Enhanced B300-like (112.5K TFLOPS, 4.5TB HBM) | DeepSeek V3, Kimi K2, Llama4 Maverick, Qwen3-235B (自研 Python event-driven simulator, 8×H100 DGX 实测验证误差<5%) | 自研 Python event-driven simulator (开源 Apache-2.0, GitHub: zhongkaiyu/waferscale_gpu_moe_sim) | Q6.1, Q6.2, Q6.6 |
| **加速器/Cerebras** | Cerebras WSE-2 (晶圆级引擎) | TSMC 7nm, 46,225mm², 2.6万亿晶体管, 850,000 AI核心；84 die 通过专有互联拼接为统一 2D mesh；每 PE 48KB scratchpad SRAM (分布式内存, 无共享内存/无硬件 cache coherence)；Weight streaming: 参数从片外 MemoryX 流到片内, 激活片上流转；数据流架构天然跳过零值计算；片上 40GB SRAM, 20 PB/s 带宽 | Cerebras CS-2 系统 | Qwen3-30B-A3B 训练, Llama 4 推理 2,522 tokens/s (BTA), MoE LLM benchmark (MoE-Inference-Bench) | Cerebras Software Platform (CGC), vLLM (CS-3 cloud inference) | Q6.1, Q6.2, Q6.6 |
| **加速器/Meta** | Meta MTIA 2i Custom NoC | 8×8 PE Array + 定制 Non-Blocking NoC Fabric；每 PE: 2×RISC-V cores + 384KB Local Memory + DPE (2×32×32B×32 MAC, 2.76 TFLOPS/s FP16/BF16) + Reduction Engine + SIMD Engine + Memory Layout Unit + Command Processor；NoC 通过 die 四侧 crossbar 连接片上共享 SRAM 和 off-chip LPDDR5 memory controllers；源端 flow control: leaky-bucket traffic shaping + packet fragmentation；3.3× BW vs MTIA 1；异步 dataflow PE 模型 | Meta MTIA 2i 芯片 (工艺节点未公开) | 推荐推理 (多模型) | 定制 RISC-V toolchain | Q6.4 |
| **加速器/PIM** | Samsung HBM-PIM | HBM Stack 内嵌入 Bank-level MAC 单元；TSV 垂直互联 (10μm pitch)；内部带宽 ~2TB/s per stack, 4× GPU HBM 有效带宽；Bank-level parallelism 并发；配合 AMD MI100 GPU 做 disaggregated inference | Samsung Aquabolt-XL HBM2-PIM + AMD MI100 GPU | — | 学术 vLLM integration | Q6.1, Q6.3 |
| **加速器/PIM** | SK Hynix GDDR6-AiM/AiMX | 16 DRAM banks, 每 bank 配备专用 MAC + AF 单元；ISR (Instruction Set Register) 指令集: MAC_ABK (16-bank 同时 256-bit MAC), MAC_SBK, WR_ABK/SBK, RD_ABK/SBK, SYNC；1.25V 低电压 (vs 标准 1.35V) 控制功耗；PCIe 加速卡形态 (2024版 32GB)；Bank-Level Parallelism 理论 256-way 并发 (16 channels × 16 banks) | SK Hynix AiMX PCIe 卡, GDDR6-AiM 芯片 | WikiText-2 perplexity (IANUS FPGA原型验证) | vLLM integration, MLIR-based compiler (PIMphony) | Q6.1, Q6.3 |
| **加速器/PIM** | UPMEM-PIM | 标准 DDR4 DIMM 形态, 20 颗 PIM 芯片/DIMM, 共 2530 PE, DPU 350MHz；计算单元 (DPU) 嵌入标准 DRAM 芯片每个 Bank 内；Host-controlled 编程模型: Host 发出 PIM 指令, PIM Controller 在 Module 内解码执行；Bank 间通信需通过 Host CPU 转发 | UPMEM-PIM DIMM (DDR4-2400) | 通用 GEMV 加速 | UPMEM SDK 2021.3.0 (clang 10.0.0) | Q6.3, Q6.5 |
| **加速器/3D NMP** | 3D Near-Memory Processing (HD-MoE/Stratum) | Hybrid Bonding (Cu-Cu 直接键合, 1μm pitch, 110,000/mm² density, ~0.88 pJ/b) 将 DRAM die 垂直堆叠在 Logic die (7nm CMOS) 之上；每个计算节点独立 local memory bank (无共享内存), 通过 2D mesh NoC (25-75 GB/s per link) 互联；Link Balance via Bayesian Optimization 最小化链路级拥塞; Dynamic Pre-Broadcast 预测 hotspot expert 提前广播 | 3D NMP 加速器 (HD-MoE: 4×4/4×8/8×8 mesh); Stratum Mono3D DRAM (1024-layer 1T1C, 8-tier memory tRCD 2.29-22.88ns) | HD-MoE: MoE all-to-all dispatch/combine, t_comm 误差<1% vs ASTRA-sim; Stratum: 8.29× throughput, 7.66× energy | HD-MoE Python 离散事件模拟器, Stratum 自建 cycle-level + system-level simulators | Q6.1, Q6.3, Q6.4 |
| **模拟器/分析级** | Timeloop / MAESTRO | 分析模型: Timeloop 通过循环嵌套映射 + 分析公式计算总访问次数和能耗, 遍历映射空间返回 Pareto-optimal 架构配置, 秒级评估; MAESTRO 通过 Roofline-style 数据流分析生成 (accelerator, dataflow, layer) 三元组的 Bandwidth Demand vs Reuse 分析, 内置 WS/OS/RS/NLR dataflow 模板, 毫秒-秒级 | 任意 DNN 加速器架构 (需手动建模) | — | Timeloop (MIT, ISPASS 2019), MAESTRO (Georgia Tech, HPCA 2019) | Q6.5 |
| **模拟器/周期级** | gem5 + Garnet / SCALEsim-v2 / DRAMsim3 / BookSim | 微架构周期级: gem5 全系统模拟 (x86/ARM/RISC-V + cache coherence + NoC), Garnet flit 级 NoC 建模 (RC→VA→SA→ST→LT pipeline, credit-based flow control, wormhole switching), 模拟速度慢 (小时级 for 全系统); SCALEsim-v2 逐 tile 逐 cycle 推进 systolic array (PE array 32×32), 分钟级; DRAMsim3 JEDEC 时序精确建模 DDR4/DDR5/HBM; BookSim 独立 NoC cycle-accurate 模拟器 (Mesh/Torus/Fat-Tree/Butterfly 拓扑) | gem5+Garnet: 4-chiplet 4×4 mesh interposer system; SCALEsim-v2: systolic array accelerator + DRAMsim3; DRAMsim3: DDR4-2400 4Gb×16 4ch 64GB/s; BookSim: 独立 NoC | gem5: PARSEC (全系统), synthetic traffic (Uniform-Random/Transpose/Bit-Rotation); SCALEsim-v2: layer-wise sparse traces (PyTorch→record); DRAMsim3: JEDEC timing verification | gem5 (多大学联合, 开源), Garnet (gem5 内置), SCALEsim-v2 (GitHub Hieu0155, MIT License), DRAMsim3 (UMD, 开源), BookSim 2.0 (Stanford, ISPASS 2013) | Q6.5, Q6.6 |
| **模拟器/事件驱动** | 自研 Event-driven Simulator / ASTRA-sim | Event-driven: 以 calibrated macro-event (COMPUTE_START/END, MEM_READ/WRITE, D2D_SEND/RECV) 替代逐 cycle 模拟, 牺牲微架构精度换取大规模可扩展性; Orders in Chaos 自研 Python simulator 处理 25-die wafer-scale + batch 16,384 tokens, 8×H100 DGX 实测验证误差<5%; RPU simulator 覆盖 36-500+ CUs DSE sweep; ASTRA-sim 建模 hierarchical network topology (intra-node NVLink + inter-node InfiniBand/RoCE), 集合通信操作级粒度 | Orders in Chaos: Tesla Dojo 5×5 + TSMC SoW 8×3; RPU: TSMC N16→N2 projection; ASTRA-sim: 分布式多节点 GPU 集群 | Orders in Chaos: DeepSeek V3, Kimi K2, Llama4, Qwen3-235B (batch 1-16384, seq 1-131K); RPU: Llama3-405B BS=1 8K seq; ASTRA-sim: DLRM, Mixtral | Orders in Chaos simulator 开源 Apache-2.0; ASTRA-sim (Georgia Tech/Intel, 开源) | Q6.5, Q6.6 |
| **RTL/EDA** | Synopsys VCS / Design Compiler / Cadence Genus / Verilator / Catapult HLS | 时钟边沿精确级: SystemVerilog/VHDL RTL 设计 → Synopsys VCS/Cadence Xcelium 功能仿真 → Synopsys Design Compiler/Cadence Genus 逻辑综合 → 面积/功耗/时序报告; Verilator 将 RTL 转为 cycle-accurate C++ 模型; Catapult HLS 从 SystemC 高层综合到 RTL; 模拟速度极慢 (OpenPiton 64-core 单 datapoint 需 5+ hours, 全曲线 >1 year) | BitMoD: TSMC 28nm; DFBM: interposer 成熟工艺 (OpenSMART BSV/Chisel RTL 生成); RPU: TSMC N16→N2 projection; Focus: Synopsys DC 28nm SS corner (0.81V, 125°C), Area: 3.21mm², Power: 736mW | BitMoD: WikiText-2/C4 perplexity; DFBM: PARSEC; Focus: Llava-Video-7B on VideoMME | Synopsys/Cadence 商业闭源; Verilator 开源; OpenSMART 开源; Catapult HLS 部分开源 | Q6.5, Q6.6 |
| **FPGA 原型** | Xilinx VCU118 / Alveo U280 / ZCU104 | FPGA 原型验证: 将关键子系统 (PIM Control Unit, PIM Memory Controllers) 综合到 FPGA, 通过 FMC 连接器连接商业 PIM 芯片 (GDDR6-AiM), 大规模部分 (NPU) 使用 cycle-accurate simulator 通过 PCIe 与 FPGA 通信; 验证指标为功能正确性 (perplexity 匹配) 而非性能 (原型频率远低于 ASIC 目标); IANUS 论文: VCU118 FPGA + GDDR6-AiM PIM + NPU Sim 混合验证, 验证 NPU-PIM 统一内存架构功能可行性 | Xilinx VCU118 (IANUS: NPU-PIM 统一内存), Xilinx Alveo U280 (Tandem Processor: 新兴算子加速器), Xilinx ZCU104 (FEATHER: 可重构 dataflow accelerator, PYNQ+Jupyter) | IANUS: GPT-2 Base/Medium/Large/XL on WikiText-2 perplexity vs full-precision; Tandem Processor: latency/throughput vs GPU baseline; FEATHER: 端到端推理延迟 vs DPU/Gemmini | Xilinx Vivado (商业); PYNQ (开源) | Q6.6 |

---

## 按方法类别分类

根据 L6 芯片设计层的覆盖范围，按以下六大类别组织：

| 分类 | 方法 | 具体方法描述 | 核心机制 | 来源 |
|------|------|-------------|----------|------|
| **Chiplet 芯粒设计** | MCM-GPU (Multi-Chiplet Module GPU) | 将 monolithic SoC 拆为多 chiplet, 通过 2.5D interposer (CoWoS/EMIB) 集成; 每 chiplet: 64 SM + L1.5 2MB + LLC slice + HBM 80GB; NUMA 效应 remote/local 延迟比 10-15×; L1.5 cache 专门缓存 remote data; 统一全局地址空间 Single-GPU-like 编程模型 | Interposer NoC + D2D Controller + NUMA-aware CTA scheduling + L1.5 remote data caching | Q6.1, Q6.2 |
| **Chiplet 芯粒设计** | DFBM (Deadlock-Free Bridge Module) | Interposer 侧 bridge module 外置 deadlock 处理; Credit-Aware Admission Control: Expected Credit Table + CVN-DB Occupancy Query 从源头阻止不可释放占用形成环依赖; 零侵入 chiplet 内部 NoC (供应商仅需提供协议参数), 拓扑无关, 标准化; 面积/功耗开销完全约束于低成本 interposer | Bridge module 外置在 interposer 侧, credit-aware admission control 阻止跨 chiplet deadlock 环形成 | Q6.1, Q6.2, Q6.4 |
| **Chiplet 芯粒设计** | Active Interposer NoC | 在 interposer 硅基板中嵌入完整 NoC 路由器 mesh 网络作为 chiplet 间共享通信背板; 每个 chiplet 通过 boundary router + TSV vertical channel 接入; 多 VN 隔离 (3 VN): 不同 coherence 消息类 (Request/Forward/Response) 分配不同 VN, 并发传输不互相阻塞; Credit-based flow control per VC; Interposer 使用成熟工艺 (65nm/28nm) 降成本 | Multi-VN 并发通道隔离 + per-VC credit flow control + XY routing on interposer mesh | Q6.1, Q6.4 |
| **Wafer-Scale 晶圆级设计** | Cerebras WSE-2 | TSMC 7nm, 46,225mm², 2.6万亿晶体管, 850,000 AI 核心; 84 die 通过专有互联拼接为统一 2D mesh (每 PE 4 邻居 32-bit bidirectional, 1 cycle/hop); 分布式内存模型 (无共享内存/无 hardware cache coherence), 每 PE 48KB scratchpad SRAM; Weight streaming: 参数从片外 MemoryX stream 到片内, 激活在片上流转; 稀疏度收割: 数据流架构天然跳过零值计算 | 2D mesh 数据流架构 + 分布式 SRAM + weight streaming + 无 cache coherence 的显式数据搬移 | Q6.1, Q6.2 |
| **Wafer-Scale 晶圆级设计** | Tesla Dojo / TSMC SoW | Dojo 5×5 2D mesh (25 dies, 每 die 1000 TFLOPS FP16, 80GB HBM, 3.35 TB/s); SoW 8×3 2D mesh (24 compute dies + 96 HBM dies, >200,000mm²); D2D: 垂直 LSI terabit-level BW + 水平 XSR SerDes 1.7 TB/s per pair, 200ns/hop; 最大 Manhattan 距离: Dojo 8 hops, SoW 10 hops; D2D 延迟模型: 7 hops × 200ns + 300ns remote HBM + 7 hops × 200ns return = 3100ns total (~10× 本地 HBM 300ns) | 2D mesh wafer 级 D2D 互联 + LSI/XSR SerDes + 每 die 独立 HBM | Q6.1, Q6.2 |
| **Wafer-Scale 晶圆级设计** | Two-Level Command Processor (两级 CP) | Global CP (wafer 级, A76-class ARM core, ~1.1mm², ~1W at 5nm): Expert Distribution Table (4.5KB SRAM, n-bit bitmask 支持 multi-die expert 复制) + Cross-token Heatmap Cache (0.5MB SRAM, 512×512 experts 512-bit width) + Task Allocation Algorithm + Data-Driven Predictor; Local CP ×25 (每 die, A72-class ARM core, ~0.3mm², ~280mW each): SM task dispatch + D2D controller config; 总面积/功耗 overhead <0.04% (6.13mm² / 8.59W per 25-die wafer) | 层次化任务分配: Global CP 运行 Task Allocation Algorithm → per-die sub-kernel 并发发送到 Local CPs → Local CPs 并发 dispatch SMs | Q6.1, Q6.2 |
| **Wafer-Scale 晶圆级设计** | Multi-Die Task Allocation (Alg.1) | Candidate Mechanism: 扩展候选 die 到邻居 die (Manhattan distance ≤1), 在 workload balance 和 D2D traffic 之间 trade-off; Block-Granularity Distribution: block size=50, 分配精度和算法开销折中; Cost Model 三维度: C_DRAM (HBM access time: local=300ns, remote=300ns+hops×200ns), C_compute (GEMM 执行时间 based on FP16 TFLOPS), C_D2D (跨 D2D links 通信时间 + central resource manager 建模 bandwidth contention); 效果: Allo Only 降 hop count 142× (vs Base), 6.3× throughput | Candidate Mechanism 扩展候选 die + Block-Granularity (block=50) + 三维 Cost Model 贪心分配 | Q6.1, Q6.2 |
| **Wafer-Scale 晶圆级设计** | Hardware-Managed HBM (ATU + PDU) | ATU (Address Translation Unit, 4.25KB SRAM, 68-bit entries, ~512 remote-to-local mappings): 第一次 remote read + caching 时动态建立地址映射; PDU (Prediction Unit, 128B register file, 16-bit entries, 每 die track 64 experts): cp_en bits 由 Global CP Data-Driven Predictor 在 kernel launch 时配置; 非传统 cache (不基于 LRU), 而是 predictor-driven selective duplication; 总面积/die ~0.0068mm², 功率 ~390mW; 完全透明于 CUDA 程序 | Predictor-driven selective duplication: PDU.is_local[e] 查询 → ATU A_remote→A_local 地址翻译 → 重定向到本地 LLC (100ns) 或 HBM (300ns) | Q6.1, Q6.2 |
| **PIM/存内计算** | DRAM-Based PIM (GDDR6-AiM/UPMEM) | Bank-Level Parallelism: 16 banks 同时执行 256-bit MAC (MAC_ABK 指令), 每 bank 从自身存储阵列读取 operand → bank-local MAC 执行乘加 → 结果写回 bank 或 Global Buffer; ISR (Instruction Set Register) 指令集: MAC_ABK/SBK, WR_ABK/SBK, RD_ABK/SBK, SYNC; 全 Bank 并发受限于 PDN 峰值供电能力 + 热密度 + DRAM cell 刷新干扰 (可能引起 row hammer); 1.25V 低电压控制功耗; 商用: AiMX PCIe 卡 32GB (2024), UPMEM DDR4 DIMM 2530 PE 350MHz | Bank-Level Parallelism + ISR 指令集 + all-bank concurrent MAC (MAC_ABK) | Q6.1, Q6.3 |
| **PIM/存内计算** | Samsung HBM-PIM | HBM Stack 的每个 DRAM Die 的 Bank I/O 路径上嵌入 MAC 计算单元 (浮点); TSV 垂直互联 (10μm pitch); 内部带宽 ~2TB/s per stack, 利用 Bank-level parallelism 提供 4× GPU HBM 有效带宽; 配合 AMD MI100 GPU disaggregated inference | Bank-level MAC 嵌入 HBM Stack + TSV 高带宽垂直互联 | Q6.1, Q6.3 |
| **PIM/存内计算** | 3D Near-Memory Processing (HD-MoE) | Hybrid Bonding (Cu-Cu 直接键合, 1μm pitch, 110,000/mm², ~0.88 pJ/b) 将 DRAM die 堆叠在 Logic die (7nm CMOS) 之上; 每个 Node 独立 local memory bank (无共享内存/无 shared L2 cache), 通过 2D mesh NoC (25-75 GB/s per link, XY routing) 互联; Compute Unit 2.5-10 TFLOPS/node; Logic die 可使用先进 CMOS 工艺独立于 DRAM 工艺节点 | Hybrid Bonding 3D 堆叠 + 2D Mesh NoC 分布式互联 + 独立 Logic die 先进工艺 + 分布式内存 | Q6.1, Q6.3 |
| **PIM/存内计算** | Duplex xPU+PIM 混合架构 | Op/B (Arithmetic Intensity) 决策调度: 每个算子计算 ArI = FLOPs/data_moved, 与硬件 Ridge Point 比较 (xPU_RP ≈ 281 Op/B, PIM_RP ≈ 8 Op/B); ArI < PIM_RP → PIM 执行 (memory-bound), ArI > xPU_RP → GPU 执行 (compute-bound); Attention 和 Expert FFN 在独立物理设备上并行执行实现算子级流水线重叠: launch_async(attention, xPU) ∥ launch_async(expert_ffn, PIM); 关键结论: 低 batch (B<32) PIM 优势明显 (MoE expert FC ArI 低, PIM 4× GPU HBM BW 优势), 高 batch (B>64) GPU 逆转 (FC ArI 超过 PIM Ridge Point) | Op/B 决策模型 + Ridge Point 硬件特征匹配 + Attention∥Expert 异构流水线并发 | Q6.3 |
| **PIM/存内计算** | ERAS (Expert Residency Aware Selection) | Routing quality vs data movement cost 权衡: Gating Logits → Softmax → Residency-Aware Adjustment (HBM 驻留 expert 概率 +α, 远端 expert 概率 -β×(1-freq[i])) → Renormalize → Top-K Selection; α=0.15 时减少 10-13% 解码延迟, offload 越多效果越显著 (最大 21.2%); 与 PIM 正交叠加: PIM 提供物理近存计算能力, ERAS 提供调度层 expert placement 优化 | Residency-Aware Gating Adjustment + 异步 HtoD transfer 流水线 | Q6.3 |
| **PIM/存内计算** | ReRAM Crossbar 存算一体 | 电阻式存储单元模拟 MVM: 输入电压向量 V[N] 施加于 crossbar 行, 各列输出电流 I_i = Σ_j (V_j × G_ij) (基尔霍夫定律+欧姆定律), 一拍完成 N×M MVM (O(1) 时间复杂度); 代表架构: PRIME (ISCA 2016, ReRAM Array 内 MVM+数字外围), ISAAC (ISCA 2016, Crossbar 内原位模拟 MVM 16-bit 定点), PipeLayer (HPCA 2017, 流水线化 MVM), FORMS (ISCA 2021, 细粒度极化原位计算), FloatPIM (ReRAM 内浮点计算), ReTransformer (ReRAM Transformer 加速); 关键约束: 模拟计算精度受限于器件 variation 和 IR-drop (需 ADC/DAC 纠错), Crossbar 写耐久度有限 (10^6-10^9 cycles, 推理友好), Sneak Current 固有噪声需补偿电路 | ReRAM Crossbar 模拟 MVM (Kirchhoff 电流求和) + O(1) 时间复杂度 + 数字外围纠错 | Q6.3 |
| **NoC/片上网络** | 2D Mesh NoC + XY Routing | 每个 tile (计算节点+内存 bank) 位于网格交叉点, 通过双向链路连接东/西/南/北四个邻居; XY routing: 先沿 X 轴后沿 Y 轴 Manhattan 最短路径, 死锁自由 (在单 NoC 内); Router pipeline: RC (Route Computation) → VA (VC Allocation) → SA (Switch Allocation) → ST (Switch Traversal) → LT (Link Traversal); 优先级队列按时间戳调度通信 chunk, link schedule dictionary 追踪链路占用; Link Balance via Bayesian Optimization 搜索逻辑集群到物理节点映射最小化链路级拥塞; 适用范围: 中小规模 (≤64 nodes, 4×4/4×8/8×8), 链路带宽 25-75 GB/s | XY routing Manhattan 最短路径 + 5-stage router pipeline + Priority Queue link scheduling + Bayesian Optimization Link Balance | Q6.1, Q6.4 |
| **NoC/片上网络** | Multi-Dimensional Torus NoC + HalfRing/DimRotation | Torus = Mesh + wrap-around 边 (维度两端相连形成环), 路径多样性优于 mesh; HalfRing 算法: 利用双向链路构建最短通信路径, hop-by-hop store-and-forward 消除多跳传输中的链路竞争; DimRotation 调度: 各维度通信序列轮转分配实现跨维度流量均衡; DOR (Dimension-Order Routing) 无故障时使用, WFR (Wild-First Routing) 链路故障时绕行; FoldedRing 容错机制; 在 TPUv3 (8×8 2D torus), TPUv4 (8×8×8 3D torus), Fugaku 超算, Amazon Trainium, Graphcore IPU-POD 广泛采用; HalfRing+DimRotation 无故障 torus 平均 2.28× 加速 (vs Google 路由 1.57×); All-to-All 占 MoE 训练 41.5%-95.7% 时间 | Wrap-around 环冗余 + HalfRing hop-by-hop store-and-forward + DimRotation 跨维度均衡 + DOR/WFR 容错路由 | Q6.4 |
| **NoC/片上网络** | Meta MTIA 2i Custom Non-Blocking NoC | 8×8 PE Array + 定制 Non-Blocking NoC Fabric: 最小化不同 initiator 间干扰; 源端 flow control: leaky-bucket traffic shaping + packet fragmentation 平滑突发流量; 每 PE 内含专用 Reduction Network 用于 PE 间结果传递 (support neighboring PE forward), 构成 coarse-grained pipeline; 两层互联: PE 内部互联 (core↔fixed-function unit 通信) + NoC (PE↔PE 和 PE↔memory 通信); Die 面积仅 +1.13× vs MTIA 1, 但综合性能提升 3×; 3.3× NoC BW vs 上一代 | Non-blocking NoC + Source-end flow control (leaky-bucket + fragmentation) + PE-level Reduction Network + 异步 dataflow PE | Q6.4 |
| **NoC/片上网络** | DFBM Inter-Chiplet NoC Deadlock Avoidance | Credit-Aware Admission Control: 追踪出口方向 credit, 从源头阻止不可释放占用形成环依赖; MTR (Turn Restriction) 替代方案: 禁止特定转向剪断 CDG 环边, 但限制路径多样性; DeFT (VC Isolation) 替代方案: Virtual Channel 划分为不重叠子图, 但每 VN 需 ≥2 VC 集成开销高; DFBM 优势: 零侵入 chiplet 内部、拓扑无关、标准化 (供应商仅需暴露协议参数); gem5+Garnet 模拟: 4-chiplet 4×4 mesh + 4×4 mesh interposer, 3 VN + per-VN 2/4 VC, XY routing, virtual cut-through flow control; DFBM latency: 1-7% reduction (avg 3%), saturation throughput: 14% 提升 vs MTR/DeFT; CVN-DB area overhead +2.5% on interposer | Credit-aware admission control + Expected Credit Table + CVN-DB occupancy query, 全开销约束于 interposer | Q6.1, Q6.4 |
| **NoC/片上网络** | ICCA All-to-All vs Mesh 拓扑 ELK 方案 | Preload Reordering: 编译时优化 execution plan, 平均编辑距离 2.9 步, 将互联拥塞降低 87.65%, 非重叠 preload 时间降至总量 0.037%; All-to-All 互联利用率 89.52% (ELK-Full) vs Mesh 57-78% (Basic); 设计洞察: HBM 和互联带宽应协同扩展 (HBM 低时增加互联无益; HBM 高时性能随互联带宽线性扩展); Compute-intensive 负载 (DiT-XL): 应优先扩展 FLOPS 而非互联/HBM; Mesh 芯片在相似推理延迟时互联利用率始终高于 all-to-all (因多跳传输) | Preload reordering (编译时) + 互联/ HBM 协同扩展 + compute-intensive vs memory-bound 差异化扩展策略 | Q6.4 |
| **芯片模拟器/工具链** | Timeloop / MAESTRO (分析级) | Timeloop: 映射空间搜索 → 分析公式计算 Access Count × Energy per level → Pareto filter → Pareto-optimal designs; MAESTRO: Dataflow roofline 分析 → Reuse = Total Ops / Total Data Access → Bandwidth Demand = Data Access / Exec Time → WS/OS/RS/NLR dataflow 对比; 建模粒度循环级/数据流级, 评估速度秒级/毫秒级; 局限: 不建模 pipeline stall/bank conflict/NoC congestion; SCAR 使用 MAESTRO 建模 MCM accelerator 各 chiplet dataflow; H2-LLM 使用 Timeloop 作为 DSE framework evaluator | 分析公式计算 + 映射空间搜索 (Timeloop) / Dataflow roofline 分析 (MAESTRO) | Q6.5 |
| **芯片模拟器/工具链** | gem5 + Garnet (周期级全系统) | 全系统周期级模拟: x86/ARM/RISC-V O3 CPU + Ruby cache coherence (MESI Two Level) + Garnet flit 级 NoC + DRAM 内存模型; DFBM 论文配置: 4 homogeneous chiplets + shared active interposer, chiplet 和 interposer 均为 4×4 mesh, XY routing, 3 VN + per-VN 2/4 VC, virtual cut-through flow control; 模拟速度: 100万指令需数秒至数分钟, 64-core 全系统 DDR5 单 datapoint >5 hours; 内部 DDR 模型精度差 (Mess 论文实测误差 52%); 对 20+ die 大规模 chiplet 系统 "prohibitively slow" | Cycle-accurate CPU + NoC (RC/VA/SA/ST/LT) + DRAM + cache coherence 联合仿真 | Q6.5, Q6.6 |
| **芯片模拟器/工具链** | Mess Simulator (分析反馈型) | 反馈控制理论驱动: CPU simulator (gem5/ZSim) 发出 memory request → Mess Feedback Controller 基于实测 BW-Latency 曲线 (从实际硬件测量获取) 调整延迟 → CPU simulator 以调整后延迟运行 window → PI controller 自适应收敛; gem5+Mess 误差仅 3% (传统 Ramulator 2 误差 52%), 速度比 Ramulator 快 13-15×; 核心优势: 不需建模内存微架构, 仅需 BW-Latency 曲线即可仿真; 局限: 无法揭示微架构瓶颈根因 | PI 控制器 + 实测 BW-Latency 曲线 as oracle + 1000-memory-op window adaptive convergence | Q6.5 |
| **芯片模拟器/工具链** | 自研 Event-Driven Simulator | 以 calibrated macro-event (COMPUTE_START/END, MEM_READ/WRITE, D2D_SEND/RECV, ALLOC_START/END) 替代逐 cycle 模拟; 每个 die 维护 compute_queue, memory_queue, d2d_link_status; 全局 contention_tracker, congestion_detector; Orders in Chaos simulator: 25-die wafer-scale + batch 16,384 tokens, 8×H100 DGX 实测校准 (单 GPU MoE expert GEMM + 两 GPU P2P 数据传输) 误差 <5%, 开源 Apache-2.0; RPU simulator: symbolic transaction (address/size/type) 替代真实 tensor data, 参数由 RTL calibrate, 覆盖 36-500+ CUs DSE sweep | Calibrated macro-event 替换逐 cycle 模拟 + per-die resource manager + 全局 contention tracker + 实测校准 (error <5%) | Q6.5, Q6.6 |
| **实现平台/实验环境** | FPGA 原型验证 (Xilinx 系列) | 混合仿真策略: 关键子系统 (PIM Control Unit + PIM Memory Controllers) 综合到 FPGA (Xilinx VCU118), 大规模部分 (NPU) 使用 cycle-accurate simulator 通过 PCIe 通信; IANUS: FPGA(PCU) → FMC → GDDR6-AiM PIM 芯片 + NPU Sim → PCIe → FPGA, 验证 PIM-NPU 统一内存架构功能可行性 (perplexity vs full-precision); Tandem Processor: Xilinx Alveo U280, 新兴算子加速器, latency/throughput vs GPU baseline; FEATHER: Xilinx ZCU104, PYNQ+Jupyter notebook, 可重构 dataflow accelerator, 端到端推理 vs DPU/Gemmini | FPGA + cycle-accurate simulator 混合验证: 小规模子系统 FPGA 原型 + 大规模计算 simulator + PCIe/FMC 互联 | Q6.6 |
| **实现平台/实验环境** | ASIC/RTL 设计与 EDA 综合 | SystemVerilog RTL 设计 → Synopsys VCS/Cadence Xcelium 功能仿真 → Synopsys Design Compiler / Cadence Genus 逻辑综合 → 面积/功耗/时序报告; BitMoD: SystemVerilog + Synopsys DC @ TSMC 28nm, cycle-level simulator 基于 RTL 综合结果校准; DFBM: OpenSMART (BSV/Chisel) 生成 NoC RTL → EDA 面积/功耗分析; RPU: SystemC + Catapult HLS → TSMC N16→N2 projection, 70-80% TDP 分配给 memory interfaces, 32 OPs/Byte; Focus: Synopsys DC 28nm SS corner (0.81V, 125°C), Area 3.21mm², Power 736mW; 多数论文未到达 tape-out 阶段 | RTL 设计 → 功能仿真 → 逻辑综合 → 面积/功耗/时序报告; RTL 仿真用于功能验证, 综合用于面积/功耗, cycle-level/event-driven simulator 用于系统级性能 | Q6.6 |

---

## 分类详细问答

### 分类: Chiplet 芯粒设计

#### 方法: MCM-GPU (Multi-Chiplet Module GPU Architecture)

将传统 monolithic SoC 拆解为多个可复用芯粒，通过先进封装（TSMC CoWoS、Intel EMIB）集成在 shared interposer 上。每个 chiplet 包含 SM 阵列（64 SM）、私有 L1 cache (128KB/SM)、chiplet 内共享 L1.5 cache (2MB)、LLC slice 和本地 DRAM partition (HBM 80GB)。所有 chiplet 的 DRAM partition 共同提供统一的全局地址空间（single logical GPU abstraction）。

- **核心机制**：
  - **Interposer NoC + D2D Controller**：Chiplet 间通过 concentrated hierarchical crossbar 或 mesh NoC 互联，典型 D2D 带宽 768 GB/s (crossbar) / 1.7 TB/s (mesh), 32 cycles/hop 或 200ns/hop
  - **NUMA 效应与缓解**：跨 chiplet 访存延迟可达本地 chiplet 的 10-15×（本地 HBM 300ns vs 远程 300ns + N×200ns/hop + 返回路径）；缓解策略：first-touch page allocation、distributed CTA scheduling、L1.5 cache 专门缓存 remote data（利用数据局部性减少跨 chiplet 访问）
  - **Cache Hierarchy**：LLC slice 仅缓存本地 DRAM data（避免 coherence 复杂度），L1.5 层专门缓存 remote data；额外 cache level 使 acquire/release 需要 invalidate/flush 更深层次，同步开销放大
  - **Chiplet 间的 MoE EP**：本质上是 "on-package EP"——expert 权重分布在各个 chiplet HBM 中，消除跨节点网络通信（NIC/IB），但引入 D2D 通信。瓶颈从 multi-GPU EP 的 inter-node network 瓶颈转化为 intra-package D2D communication 瓶颈
  - **良率经济学**：小 die 良率远高于大 die（Poisson 模型: Y_die = e^(-D·A_die)），chiplet 通过 KGD (Known-Good-Die) 策略在高良率下实现大规模集成。代价是引入 C_interposer 和 C_assembly，以及 Y_assembly < 1

- **实现**：
  - 商业产品: AMD MI300X (8 compute chiplets), NVIDIA Blackwell B200 (2 reticle-limited die via NV-HBI 10 TB/s), NVIDIA Rubin (规划 4 chiplets)
  - Interposer 工艺: 65nm 成熟工艺（布线密度亚微米线宽），TSV + microbump
  - 编程模型: Single-GPU-like（Blackwell/Rubin 方向），CUDA 程序无需修改

- **实验环境**：
  - 模拟器: gem5 + Garnet (cycle-accurate NoC), 自研 Python event-driven simulator
  - Benchmark: PARSEC, synthetic traffic (Uniform-Random/Transpose/Bit-Rotation), MoE 模型 (DeepSeek V3, Kimi K2)
  - 硬件平台: NVIDIA B200, AMD MI300X, H100 DGX (实测验证)
  - 关键指标: D2D BW 768 GB/s-10 TB/s, 32 cycles/hop, NUMA ratio 10-15×, 良率 ~82%/die (200mm², D=0.1/cm²)

- **来源**: Q6.1 (MCM-GPU Architecture note score 143), Q6.2 (MCM-GPU Architecture note score 375.2, DFBM VII Evaluation score 2674.0), vault: `knowledge_notes/芯片知识笔记/Multi-Chiplet GPU (MCM-GPU) Architecture _ 多芯粒GPU架构.md`, `knowledge_notes/芯片知识笔记/2.5D Silicon Interposer (2.5D 硅中介层).md`

---

#### 方法: DFBM (Deadlock-Free Bridge Module) — Chiplet 间 NoC 死锁避免

即使单个 chiplet 内部 NoC 已 deadlock-free，当多个 chiplet 通过 active interposer 互连后，chiplet NoC 与 interposer NoC 之间可能形成跨 chiplet 的 cyclic channel dependency。DFBM 将 deadlock 处理从 chiplet 内部外移到 interposer 侧 bridge module。

- **核心机制**：
  - **Credit-Aware Admission Control**：追踪出口方向 credit，从源头阻止不可释放占用形成环依赖——Expected Credit Table + CVN-DB Occupancy Query
  - **与替代方案对比**：
    - MTR (Modular Turn Restriction): 边界路由器转向限制，但负载不均、非拓扑无关
    - DeFT: 专用 VC 隔离上下行流量，拓扑无关但每 VN 需 ≥2 VC，集成开销高
    - UPP/Steered Bubble: 允许死锁发生再恢复，高资源利用率但需死锁检测逻辑
    - DFBM: 零侵入 chiplet 内部、拓扑无关、标准化（chiplet 供应商只需提供协议参数: coherence state machine 依赖、最大 outstanding request 数、VC 数量），全开销约束于低成本 interposer
  - **CDG (Channel Dependency Graph) 理论**：NoC 死锁的充要条件是 CDG 有环。多 chiplet NoC 的死锁环可跨越 chiplet 内部 NoC 和 interposer NoC 的边界

- **实现**：OpenSMART (BSV/Chisel) 生成 NoC RTL, EDA 工具进行面积/功耗分析, DFBM module 位于 interposer 侧

- **实验环境**：
  - gem5 + Garnet: 4 homogeneous chiplets + shared active interposer, 各为 4×4 mesh, XY routing, 3 VN + per-VN 2/4 VC, virtual cut-through flow control
  - Synthetic traffic: Uniform-Random/Transpose/Bit-Rotation; Full-system: x86 Linux + PARSEC benchmark
  - 关键指标: latency 1-7% reduction (avg 3%), saturation throughput 14% 提升 (vs MTR/DeFT), CVN-DB area +2.5% on interposer

- **来源**: Q6.1, Q6.2, Q6.4; vault: `paper_secs/secs_2026/61-Deadlock-Free Bridge Module.../VII.-EVALUATION.md` (score: 3884.7), `knowledge_notes/芯片知识笔记/Channel Dependency Graph (CDG) for Multi-Chiplet NoC _ 多芯粒NoC的通道依赖图.md` (score: 201)

---

#### 方法: Active Interposer NoC — 多算子并发通道隔离

Active interposer 嵌入有源逻辑电路（NoC 路由器、VC buffer、credit 管理），构成 chiplet 间共享通信基础设施。每个 chiplet 通过 boundary router + vertical channel (TSV-based) 接入 interposer NoC。

- **核心机制**：
  - **并发通道隔离**：不同 coherence 消息类（Request/Forward/Response）分配到不同 VN（最少 3 个 VN），在 interposer NoC 上并发传输不互相阻塞
  - **Credit-based flow control**：每 VC 独立 credit 管理，避免一个 chiplet 的慢速消费阻塞其他 chiplet 的通信
  - **DFBM shared deadlock buffer**：当多个 chiplet 同时发起跨 chiplet 通信时，bridge module 协调 credit 分配和缓冲管理
  - **UCIe 标准**：Universal Chiplet Interconnect Express 定义 die-to-die physical layer 标准

- **实现**：Interposer 使用 logic process (65nm/28nm)，路由器和 VC buffer 的面积/功耗由成本较低的 interposer 侧承担

- **实验环境**：gem5 + Garnet 全系统模拟, 典型配置 4×4 mesh topology + XY routing + 3 VN + per-VN 2/4 VC + virtual cut-through flow control

- **来源**: Q6.1, Q6.4; vault: `knowledge_notes/芯片知识笔记/Active Interposer _ 主动中介层.md` (score: 65/637.4)

---

### 分类: Wafer-Scale 晶圆级设计

#### 方法: Cerebras WSE-2 (Wafer-Scale Engine)

在整片 300mm 硅晶圆上集成计算核心，84 个 die 通过专有互联技术拼接在单晶圆上构成统一 2D mesh 网络。TSMC 7nm, 46,225mm², 2.6 万亿晶体管, 850,000 AI 核心, 40GB on-chip SRAM, 20 PB/s 内存带宽。

- **核心机制**：
  - **分布式内存模型**：无共享内存、无硬件 cache coherence。每个 PE 有 48KB scratchpad SRAM，所有数据传输由软件显式管理
  - **数据流架构**：weight streaming 模式将参数存储与计算解耦，权重从片外 MemoryX stream 到片内，激活在片上流转
  - **稀疏度收割**：数据流架构天然跳过零值计算
  - **2D mesh 互联**：每 PE 通过 32-bit 双向端口连接 4 个邻居 (N/S/E/W)，单跳延迟 1 cycle
  - **冗余容错**：850,000 核心中预留大量 spare cores + reconfigurable interconnect 绕过缺陷区域——接受缺陷，通过冗余容错

- **实现**：Cerebras CS-2 系统，Cerebras Software Platform (CGC), vLLM (CS-3 cloud inference)

- **实验环境**：
  - 真实硬件 CS-2/CS-3 系统
  - MoE LLM benchmark: Qwen3-30B-A3B 训练 (BTA), Llama 4 推理 2,522 tokens/s
  - 物理对比 (vs H100): 面积 57×, 片上内存 800× (40GB vs 0.05GB L2), 带宽 10,000× (20 PB/s vs 0.002 PB/s L2), 核心数 56×

- **来源**: Q6.1, Q6.2; vault: `knowledge_notes/芯片知识笔记/Wafer-Scale Engine (WSE _ 晶圆级引擎).md` (score: 1613.7), `experiment_notes/硬件实验笔记/MoE-Inference-Bench_系统实验笔记.md` (score: 26.9)

---

#### 方法: Tesla Dojo / TSMC SoW — Wafer-Scale Multi-Chiplet GPU 拓扑

Dojo: 5×5 2D mesh (25 dies, 每 die 1000 TFLOPS FP16, 80GB HBM, 3.35 TB/s); SoW: 8×3 2D mesh (24 compute dies + 96 HBM dies, >200,000mm²)。垂直方向 LSI terabit-level BW，水平方向 XSR SerDes 1.7 TB/s per adjacent die pair, 200ns/hop。

- **核心机制**：
  - **D2D 延迟模型**：以 Die0 访问 Die7 expert 数据为例: 7 hops × 200ns + 300ns remote HBM + 7 hops × 200ns return = 3100ns total (~10× 本地 HBM 300ns)
  - **方形 vs 矩形拓扑差异**：Dojo 5×5 (最大 Manhattan 8 hops) vs SoW 8×3 (最大 Manhattan 10 hops)，SoW 矩形布局 baseline hop count 更高，优化空间更大 (Allo+Pred: 7.5× on SoW vs 7.0× on Dojo)
  - **Dojo-Enhanced (B300-like)**: 4500 TFLOPS FP16, 180GB HBM, 8 TB/s BW per die, D2D 2 TB/s。GPU 性能 outpace interconnect bandwidth (TFLOPS 4.5× but D2D BW 仅 1.18×)，更凸显 on-GPU-command-processor 必要性——Host CPU 实现 overhead 达 42-51.6%

- **实现**：自研 Python event-driven multi-chiplet GPU simulator (开源 Apache-2.0, GitHub: zhongkaiyu/waferscale_gpu_moe_sim)

- **实验环境**：
  - 四种 MoE 模型: DeepSeek V3 (671B), Kimi K2 (1T), Llama4 Maverick, Qwen3-235B
  - 8×H100 DGX 实测校准 (单 GPU MoE expert GEMM + 两 GPU P2P 数据传输), 误差 <5%
  - 关键指标: 6.3-7.5× throughput, 142-213× hop count reduction, <0.04% area overhead

- **来源**: Q6.1, Q6.2; vault: `knowledge_notes/芯片知识笔记/System-on-Wafer (SoW) Technology (晶圆级系统集成技术).md` (score: 2334.4), `knowledge_notes/硬件知识笔记/Wafer-Scale Multi-Chiplet GPU for MoE Serving (晶圆级多芯粒GPU的MoE推理).md` (score: 1628.1)

---

#### 方法: Two-Level Command Processor (两级命令处理器)

传统 GPU 单层 CP 将所有 SM 视为均等资源（忽略物理位置和 data placement），在 wafer-scale 场景下导致大量不必要 D2D traffic 和严重负载不均。两级 CP 重新设计 Command Processor 实现 expert-placement-aware 任务分配。

- **核心机制**：
  - **Global CP** (wafer 级, A76-class ARM core, ~1.1mm², ~1W at 5nm): Expert Distribution Table (4.5KB SRAM, n-bit bitmask 而非 single die ID——支持 expert 被复制到多个 die) + Cross-token Heatmap Cache (0.5MB on-chip SRAM, 512×512 experts 512-bit width) + Task Allocation Algorithm (Alg.1) + Data-Driven Predictor
  - **Local CP ×25** (每 die, A72-class ARM core, ~0.3mm² each, ~280mW each at 5nm): SM task dispatch + D2D controller config
  - **Kernel Launch 并发工作流**：Host CPU → Global CP: "Launch MoE kernel for layer l" → Global CP 读 expert_reqs_dict → 运行 Task Allocation Algorithm → Predictor 计算 cp_en bits → 打包 sub-kernel descriptor → 并发发送到所有 Local CPs via D2D → 每个 Local CP 并发分配 sub-kernel 到本 die SMs + 配置 PDU prediction table → 等待 SM completion → 并发报告 expert duplication statistics → Global CP 更新 Expert Distribution Table
  - 总面积/功耗 overhead <0.04% (6.13mm² / 8.59W per 25-die wafer, 5nm)

- **实现**：Register files: Yosys synthesis; SRAM: CACTI modeling; 均 scaled to 5nm (H100 process node)

- **实验环境**：Orders in Chaos 自研 Python event-driven simulator, 四种 MoE 模型 + 两种拓扑, 8×H100 DGX 实测验证

- **来源**: Q6.1, Q6.2; vault: `knowledge_notes/硬件知识笔记/Two-Level Command Processor (两级命令处理器).md` (score: 278.8)

---

#### 方法: Multi-Die Task Allocation (Alg.1)

运行在 Global CP 上的启发式算法，将 MoE kernel 计算按 expert 拆分为 per-die 子任务，支持多 die 并发执行。运行频率：每 MoE layer kernel launch 时执行一次（非 per-token）。

- **核心机制**：
  - **Candidate Mechanism**：扩展候选 die 到邻居 die (Manhattan distance ≤1)，而非仅限于 expert 所在 die。在 workload balance 和 D2D traffic 之间 trade-off——传统 EP 将所有请求分配到本地 die 避免 D2D 但导致热门 expert 所在 die 严重过载 (16× 平均请求量)
  - **Block-Granularity Distribution**：block size=50，分配精度和算法开销折中
  - **Cost Model 三维评估**：C_DRAM（读取 expert 权重的 HBM access time: local=300ns, remote=300ns+hops×200ns）、C_compute（基于 die FP16 TFLOPS 估算 GEMM 执行时间）、C_D2D（跨 D2D links 通信时间 + central resource manager 建模 bandwidth contention）
  - **效果**：Allo Only 降 hop count 142× (vs Base), 6.3× throughput; Allo+Pred 降 hop count 213×, 7.0-7.5× throughput

- **实现**：Global CP 上运行的启发式算法，Python event-driven simulator 中实现

- **实验环境**：自研 simulator, 四种 MoE 模型 (DeepSeek V3, Kimi K2, Llama4, Qwen3), 两种拓扑 (Dojo 5×5, SoW 8×3)

- **来源**: Q6.1, Q6.2; vault: `knowledge_notes/kernel知识笔记/Multi-Die Task Allocation for MoE (多Die MoE任务分配).md` (score: 1691.2)

---

#### 方法: Hardware-Managed HBM (ATU + PDU)

在 D2D controller 中集成 ATU 和 PDU，实现对远程 HBM 中热门 expert 权重的硬件自动本地缓存。软件/CUDA 程序完全无感知。

- **核心机制**：
  - **ATU** (Address Translation Unit, 4.25KB SRAM, 68-bit entries, ~512 remote-to-local mappings): 第一次 remote read + caching 时动态建立地址映射 (A_remote→A_local)，由 D2D controller 硬件自动执行
  - **PDU** (Prediction Unit, 128B register file, 16-bit entries, 每 die 最多 track 64 experts): cp_en bits 由 Global CP 在 kernel launch 时通过 Data-Driven Predictor 计算并配置
  - **并发数据流路径**：Case 1 (Remote Read → 未缓存): SM → D2D Controller → check PDU.is_local[e]=0 → 正常 D2D read (multi-hop XY routing) → remote die HBM → return data → 若 PDU.cp_en[e]=1 则写入本地 HBM + ATU 建立映射；Case 2 (Local Read → 已缓存): SM → D2D Controller → PDU.is_local[e]=1 → ATU A_remote→A_local → 重定向到本地 LLC (100ns hit) 或本地 HBM (300ns) ——完全避免 D2D 延迟
  - 不是传统 cache (不基于 LRU)，而是 predictor-driven selective duplication
  - 总面积/die ~0.0068mm², 功率 ~390mW；可推广到 CXL-based systems、SSD offloading、PIM systems

- **实现**：D2D Controller 内硬件模块，Register files + SRAM, scaled to 5nm

- **实验环境**：Orders in Chaos 自研 simulator, 8×H100 DGX 实测验证

- **来源**: Q6.1, Q6.2; vault: `knowledge_notes/硬件知识笔记/Hardware-Managed HBM with ATU and PDU (硬件管理的HBM缓存).md` (score: 245.7)

---

### 分类: PIM/存内计算

#### 方法: DRAM-Based PIM — Bank-Level Parallelism (GDDR6-AiM / UPMEM)

将计算单元（MAC、ALU）嵌入 DRAM 芯片 Bank 内部或 Bank I/O 路径，利用 DRAM 内部极高带宽 (16-32 TB/s 级别) 直接在内存中执行计算。

- **核心机制**：
  - **SK Hynix GDDR6-AiM**：16 DRAM banks, 每 bank 配备专用 MAC + AF 单元。ISR (Instruction Set Register) 指令集——MAC_ABK: All-Bank MAC (16 banks 同时 256-bit MAC, 每个 Bank 从自身存储阵列读取 operand → bank-local MAC 执行乘加 → 结果写回 bank 或 Global Buffer), MAC_SBK: Single-Bank MAC (低并发/功耗受限), WR_ABK/SBK: 写入操作数到 Bank (buffer preload), RD_ABK/SBK: 从 Bank 读取计算结果, SYNC: Barrier 同步协调跨 Bank 数据依赖
  - **全 Bank 并发约束**：PDN 峰值供电能力 (16 Bank 同时 active 远超标准 DRAM 功耗预算), 热密度, DRAM cell 刷新干扰 (相邻 Bank 同时 active 可能引起 row hammer), tFAW 限制需放宽 (影响 DRAM cell data retention margin)
  - **UPMEM-PIM**：标准 DDR4 DIMM, 20 颗 PIM 芯片 (2530 PE, DPU 350MHz), 计算单元嵌入每个 Bank 内部, Host 发出 PIM 指令, PIM Controller 在 Module 内解码执行。Bank 间通信必须通过 Host CPU 转发
  - **数据搬运消除核心**：Expert 权重驻留 PIM Bank 本地 (无需搬运), 仅传输 hidden states 向量 (h_dim 大小) 到 PIM Global Buffer, 搬运量从 O(expert_size) 降低到 O(hidden_states)

- **实现**：
  - SK Hynix AiMX: PCIe 加速卡 (2024版 32GB), 1.25V 低电压 (vs 标准 GDDR6 1.35V), 软件栈支持 vLLM 框架, 实现与 NVIDIA H100 disaggregated inference (2×H100 + 4×AiMX)
  - UPMEM: UPMEM SDK 2021.3.0 (clang 10.0.0), DDR4-2400
  - PIMphony: MLIR-based compiler (IREE runtime HAL 对接 PIM SDK)

- **实验环境**：
  - 模拟器: Ramulator 2.0 + AiM Simulator (cycle-accurate DRAM command 级), PIMphony Simulator (MLIR + Ramulator), UPMEM SDK (真实硬件)
  - 商用部署: AiMX PCIe 加速卡, Samsung HBM-PIM 配合 AMD MI100 GPU
  - 关键指标: 1 TFLOPS/chip (AiMX), 16-32 TB/s internal BW, 数据搬运减少 ~77-99.99% (取决于算子和 batch size)

- **来源**: Q6.1, Q6.3; vault: `knowledge_notes/芯片知识笔记/DRAM-Based Processing-in-Memory (PIM).md` (score: 5960.6), `knowledge_notes/硬件知识笔记/AiMX PIM Architecture.md` (score: 788.1), `paper_secs/.../PIM-MMU.md` (score: 14676.9)

---

#### 方法: Duplex xPU+PIM 混合架构 — Op/B 决策调度

将 Attention 和 Expert FFN 分配到不同物理设备 (GPU 和 PIM) 并发执行，通过 Op/B (Arithmetic Intensity) 决策模型实现最优算子-设备匹配。

- **核心机制**：
  - **Ridge Point 决策**：每个算子计算 ArI = FLOPs / data_moved，与硬件 Ridge Point 比较 (xPU_RP ≈ 281 Op/B, PIM_RP ≈ 8 Op/B)。ArI < PIM_RP → PIM 执行 (memory-bound), ArI > xPU_RP → GPU 执行 (compute-bound)
  - **异构并发流水线**：launch_async(attention_kernel, xPU_device) ∥ launch_async(expert_ffn_kernel, PIM_device)，两者在独立物理设备上执行，无需争抢同一内存带宽。总延迟 ≈ max(Attn_time, Exp_time) + TSV_sync_overhead。MoE 场景典型 Attn_time ≈ Exp_time → ~2× 并发加速
  - **Batch Size 效应 (关键约束)**：
    - 低 batch (B < 32): PIM 优势明显——MoE expert FC 层 ArI 低，PIM 的 4× GPU HBM 带宽优势使执行时间更短
    - 高 batch (B > 64): GPU 逆转——大 batch 使 FC 层 ArI 超过 PIM Ridge Point (8 Op/B)，GPU 以 RP_acc=281 提供更高计算吞吐
    - MLA + MoE 架构下大 batch 推理已成主流 (MLA 释放了 KV cache 约束)，PIM 仅适用于低 batch/低序列长度的推理场景

- **实现**：Host-controlled 编程模型 (Host 发出 PIM 指令, PIM Controller 在 Module 内解码执行), xPU 和 PIM 通过 TSV 高带宽通道交换中间结果

- **实验环境**：论文自研微架构模拟器 (HBM-based PIM bank 级建模), 未公开开源

- **来源**: Q6.3; vault: `knowledge_notes/硬件知识笔记/PIM (Processing-in-Memory) for MoE.md` (score: 5776.7), `knowledge_notes/硬件知识笔记/Processing-in-Memory (PIM) for LLM MoE Inference.md` (score: 5050.2)

---

#### 方法: 3D Near-Memory Processing — HD-MoE/Stratum

通过 Hybrid Bonding 将 DRAM die 垂直堆叠在 Logic die (7nm CMOS) 之上，每个计算节点拥有独立 local memory bank (无共享内存)，通过 2D mesh NoC 互联。

- **核心机制**：
  - **Hybrid Bonding**：Cu-Cu 直接键合, 1μm pitch, 110,000/mm² density, ~0.88 pJ/b (远优于传统 PCB 互连)
  - **与 GPU 关键区别**：无 shared L2 cache、无 global memory、分布式 bank-local memory、有限 NoC 带宽 (25-75 GB/s per link, 非 NVLink/NVSwitch)
  - **MoE Token Dispatch/Combine 并发执行**：Phase 1 Token Dispatch (all-to-all) ——所有 token dispatch 同时进行 (受 NoC link BW 限制); Phase 2 Expert FFN Compute ——所有节点 expert 计算完全并行 (无跨节点依赖), Intra-node 并发: gate_proj/up_proj/down_proj 可流水线; Phase 3 Result Combine (all-to-all) ——所有 combine 消息并发传输
  - **HD-MoE Link Balance**：Bayesian Optimization 搜索逻辑集群到物理节点的映射，最小化 NoC 链路级拥塞
  - **Dynamic Pre-Broadcast**：利用 expert activation 的时间局部性预测 hotspot expert → 提前 broadcast 到所有节点 → token 路由时选择已有 expert 副本的最低负载节点
  - **HD-MoE TP/EP 混合并行**：Hot Expert 采用 TP (多节点协作计算, 负载均衡), Cold Expert 采用 EP (单节点本地计算, 零通信)。LP+BO 优化实现 1.1-1.8× vs TP, 1.1-1.5× vs EP
  - **Stratum Mono3D DRAM**：1024-layer 1T1C, 8-tier memory (tRCD 2.29-22.88ns 动态分级), Topic-Aware Placement, 8.29× throughput, 7.66× energy

- **实现**：Logic die 独立于 DRAM 工艺节点 (7nm CMOS), Cadence Genus + FinCACTI (Stratum), 自研 Python 离散事件模拟器 (HD-MoE, ASTRA-sim 交叉验证 ring all-reduce 误差 <1%)

- **实验环境**：HD-MoE: 2D mesh 4×4/4×8/8×8, Node 级建模 (compute unit, DRAM bank, NoC link), 25-75 GB/s NoC link, 2.5-10 TFLOPS/node; Stratum: Mono3D DRAM + PU groups, MoE throughput/latency benchmark

- **来源**: Q6.1, Q6.3; vault: `knowledge_notes/硬件知识笔记/3D Near-Memory Processing (3D NMP) for MoE LLM Inference.md` (score: 9244/1466.0), `idea_notes/HD-MoE with 3D NMP.md` (score: 537.2), `idea_notes/Stratum Mono3D DRAM.md` (score: 133.7)

---

#### 方法: ReRAM Crossbar 存算一体

利用电阻式存储单元的模拟特性直接在 Crossbar 阵列中完成矩阵向量乘法 (MVM)。输入电压向量 V[N] 施加于 crossbar 行, 各列输出电流 I_i = Σ_j (V_j × G_ij) (基尔霍夫定律 + 欧姆定律), 一拍完成 N×M MVM (O(1) 时间复杂度)。

- **核心机制**：
  - **模拟 MVM 计算**：Crossbar 内原位模拟计算——所有权重存储在 ReRAM cell conductance G_ij 中，MVM 通过 Kirchhoff 电流求和在一拍内完成
  - **关键设计约束**：(1) 模拟计算精度受限于器件 variation 和 IR-drop，通常需要数字外围电路进行纠错 (ADC/DAC 开销大)；(2) Crossbar 写耐久度有限 (通常 10^6-10^9 cycles)，训练场景不适用但推理场景权重静止非常适配；(3) Sneak Current 是 Crossbar 阵列的固有噪声源，需要补偿电路
  - **代表架构**：PRIME (ISCA 2016, ReRAM Array 内 MVM+数字外围), ISAAC (ISCA 2016, Crossbar 内原位模拟 MVM 16-bit 定点), PipeLayer (HPCA 2017, 流水线化 ReRAM MVM), FORMS (ISCA 2021, 细粒度极化 ReRAM 原位计算), FloatPIM (ReRAM 内浮点计算 FP32), ReTransformer (ReRAM Transformer 加速)

- **实现**：ReRAM Array + ADC/DAC + 数字外围纠错电路

- **实验环境**：笔记未明确说明 ReRAM Crossbar 有标准化的开源评估框架；笔记中 ReRAM 相关信息主要出现在参考文献中，缺乏 MoE/DiT/多模态/Video 具体场景下 ReRAM 集成的实验数据

- **来源**: Q6.3; vault: `paper_secs/.../REPA/References.md`, `paper_secs/.../25-Be CIM.../References.md`

---

### 分类: NoC/片上网络

#### 方法: 2D Mesh NoC + XY Routing for MoE All-to-All

2D mesh 是 chiplet 间、3D NMP 节点间和 wafer-scale die 间最广泛使用的 NoC 拓扑。每个 tile 位于网格交叉点，通过双向链路连接四个邻居 (E/W/N/S)。XY routing 先沿 X 轴后沿 Y 轴，Manhattan 最短路径，死锁自由 (在单 NoC 内)。

- **核心机制**：
  - **Router Pipeline**：RC (Route Computation) → VA (Virtual-Channel Allocation) → SA (Switch Allocation) → ST (Switch Traversal) → LT (Link Traversal)，cycle-accurate 模拟
  - **Priority Queue Link Scheduling**：通信任务按 chunk 切分进入 priority queue 按时间戳调度；link schedule dictionary 追踪每条链路占用时间表，新任务仅在链路空闲时传输；传输时间 = chunk_size / BW
  - **XY routing 路径预缓存**：避免重复计算，离散事件模拟器验证误差 <1% (vs ASTRA-sim: 模型预测 673µs vs 仿真 668µs)
  - **Intra-Expert (TP all-reduce) 和 Inter-Expert (EP all-to-all) 通信**：两类模式在同一 2D mesh NoC 上并发
  - **Node-Link Balance 协同优化**：Phase 1 (Node Balance via LP) 平衡计算负载和通信量；Phase 2 (Link Balance via Bayesian Optimization) 最小化链路级拥塞
  - **适用范围**：中小规模 (≤64 nodes, 4×4/4×8/8×8), 链路带宽 25-75 GB/s (3D NMP) 或 1.7 TB/s (D2D)

- **实现**：HD-MoE: 自研 Python 离散事件模拟器 (ASTRA-sim 交叉验证 <1%); gem5 + Garnet: cycle-accurate NoC 模拟 (RC/VA/SA/ST/LT pipeline, credit-based flow control)

- **实验环境**：HD-MoE 2D mesh 模拟器: Node 级建模, 4×4/4×8/8×8 mesh, 25-75 GB/s links; gem5+Garnet: 4 chiplet + interposer 各 4×4 mesh, PARSEC + synthetic traffic

- **来源**: Q6.1, Q6.4; vault: `paper_secs/secs_moe/HD-MoE.../A.-Overview.md` (score: 4073.7), `knowledge_notes/硬件知识笔记/2D Mesh NoC in 3D Near-Memory Processing.md` (score: 7855/39.4)

---

#### 方法: Multi-Dimensional Torus NoC + HalfRing/DimRotation

Torus = Mesh + wrap-around 边 (维度两端相连形成环)，路径多样性优于 mesh。广泛用于 TPUv3 (8×8 2D torus)、TPUv4 (8×8×8 3D torus)、Fugaku 超算、Amazon Trainium、Graphcore IPU-POD。

- **核心机制**：
  - **HalfRing 算法**：利用双向链路构建最短通信路径，hop-by-hop store-and-forward 消除多跳传输中的链路竞争
  - **DimRotation 调度**：各维度通信序列轮转分配，实现跨维度流量均衡
  - **容错路由**：DOR (Dimension-Order Routing) 无故障时使用，WFR (Wild-First Routing) 链路故障时绕行，FoldedRing + MATE 容错机制
  - **All-to-All 瓶颈量化**：占 MoE/DLRM 训练时间 41.5%-95.7%，HalfRing+DimRotation 无故障 torus 平均 2.28× 加速 (vs Google 路由方案 1.57×), 故障时 FoldedRing+MATE 平均 1.37× over fault-free baseline

- **实现**：ASTRA-sim (analytical + GARNET backend), 分布式 DL 通信模拟

- **实验环境**：ASTRA-sim TPUv3/v4 拓扑建模, DLRM + Mixtral Top-2 MoE benchmark

- **来源**: Q6.4; vault: `paper_secs/secs_moe/Optimizing All-to-All...Torus Networks/1-Introduction.md` (score: 1569.2)

---

#### 方法: Meta MTIA 2i Custom Non-Blocking NoC

8×8 PE Array + 定制 Non-Blocking NoC Fabric，最小化不同 initiator 间干扰。源端 flow control: leaky-bucket traffic shaping + packet fragmentation 平滑突发流量防拥塞。

- **核心机制**：
  - **PE 内部异步 dataflow 模型**：RISC-V cores 生成 custom instructions → fixed-function units 按依赖关系异步执行
  - **每 PE 内含**：2×RISC-V cores (1 with 64B SIMD) + 384KB Local Memory + DPE (2×32×32B×32 MAC, 2.76 TFLOPS/s FP16/BF16) + Reduction Engine (专用 Reduction Network 支持 neighboring PE forward 构成 coarse-grained pipeline) + SIMD Engine (FP16/BF16/FP32 ALUs + LUTs) + Memory Layout Unit (Transpose/Concat/Reshape) + Command Processor (Dependency Check + Scheduling) + Fabric Interface (DMA to NoC)
  - **两层互联**：PE 内部互联 (core↔fixed-function unit 通信) + NoC (PE↔PE 和 PE↔memory 通信)
  - **NoC 通过 die 四侧 crossbar** 连接片上共享 SRAM 和 off-chip LPDDR5 memory controllers; DPE-to-Local Memory 带宽翻倍, FI-to-NoC 带宽翻倍 (vs MTIA 1)
  - Die 面积仅 +1.13× (vs MTIA 1), 综合性能提升 3×, 3.3× NoC BW 提升

- **实现**：定制 RISC-V toolchain, 芯片 die 物理实现 (工艺节点未公开)

- **实验环境**：多模型推荐推理 benchmark, MTIA 1 vs MTIA 2i 对比

- **来源**: Q6.4; vault: `paper_secs/.../48-Meta_s Second Generation AI Chip.../Metas-Second-Generation-AI-Chip...md` (score: 1156.2)

---

#### 方法: ICCA All-to-All vs Mesh 拓扑 ELK 方案

编译时 Preload Reordering 消除互联拥塞：基于通用 expert 形状优化 execution plan (平均编辑距离 2.9 步)，运行时根据 gate 选择的 expert index 加载具体 expert tensor。

- **核心机制**：
  - **Preload Reordering**：消除 87.65% 互联拥塞开销，非重叠 preload 时间降至总量 0.037%
  - **All-to-All vs Mesh 互联利用率**：ELK-Full All-to-All 89.52% vs Mesh 57-78% (Basic)
  - **协同扩展策略**：HBM 低时增加互联带宽无益 (HBM 是瓶颈)；HBM 高时性能随互联带宽线性扩展
  - **按负载差异化扩展**：Compute-intensive (DiT-XL) 优先扩展 FLOPS；Memory-bound (LLM decode) 优先扩展互联/HBM；512+ cores 时 preload 优化收益递减
  - **非 GQA 模型影响**：Llama2-13B, OPT-30B 等非 GQA 模型在 mesh 上 KV cache 多跳传输导致更高拥塞

- **实现**：ELK ICCA Chip Emulator, 互联/HBM/计算联合模拟

- **实验环境**：Llama2-7B/13B/70B, OPT-30B, Gemma2-27B, DiT-XL, 512-2048 cores config

- **来源**: Q6.4; vault: `paper_secs/.../12-ELK.../span-idpage-9-1span6.2-End-to-end-Performance.md` (score: 1578.9)

---

### 分类: 芯片设计模拟器与工具链

#### 方法: Timeloop — 数据流架构 DSE

MIT (Parashar et al., ISPASS 2019)，分析模型，建模粒度循环级。将 DNN 层的 loop nest 映射到硬件加速器的存储层级 (DRAM→Global Buffer→Local Buffer→RF→MAC) 和数据流策略 (WS/OS/RS/NLR)，以能耗或延迟为优化目标遍历映射空间返回 Pareto-optimal 架构配置。

- **核心机制**：
  - **映射空间搜索**：loop permutation + tiling factor + spatial unrolling, 遍历所有映射候选
  - **分析评估公式**：T_total = Σ(Accesses_l × Energy_l / Bandwidth_l) + ComputeCycles
  - **H2-LLM 使用**：作为 DSE framework 核心 evaluator，搜索 heterogeneous hybrid-bonding accelerator 最优 dataflow
- **局限**：不建模 pipeline stall/bank conflict/NoC congestion
- **评估速度**：秒级
- **来源**: Q6.5; vault: `paper_secs/.../TAIDL.../span-idpage-12-3span8.2-Case-Study...md` (score: 525.1), `paper_secs/.../H2-LLM.../Hsup2sup-LLM...md` (score: 1250.7)

---

#### 方法: gem5 + Garnet — 全系统周期级模拟

多大学联合开发，全系统周期级模拟器。支持 x86/ARM/RISC-V O3 CPU + Ruby cache coherence (MESI Two Level) + Garnet flit 级 NoC (RC→VA→SA→ST→LT pipeline, credit-based flow control, wormhole switching) + DRAM 内存模型。

- **核心机制**：
  - **DFBM 论文配置示例**：4 homogeneous chiplets + shared active interposer, chiplet 和 interposer 各 4×4 mesh, XY routing, 3 VN + per-VN 2/4 VC, virtual cut-through flow control
  - **评估输出**：latency histogram, throughput, NoC link utilization
- **关键局限**：模拟速度慢 (100 万指令需数秒至数分钟，64-core 全系统 DDR5 单 datapoint >5 hours); 内部 DDR 模型精度差 (Mess 论文实测误差 52% vs 实际 Graviton3 服务器); 对 20+ die 大规模 chiplet 系统 "prohibitively slow"
- **适用范围**：Chiplet NoC 死锁分析 (gem5+Garnet)、PIM 评估 (gem5 SE mode + CACTI)、内存系统评估 (gem5 + multiple memory models)
- **评估速度**：小时级 (全系统)
- **来源**: Q6.5, Q6.6; vault: `paper_secs/.../DFBM/VII.-EVALUATION.md` (score: 2519.6/3884.7), `paper_secs/.../MIMDRAM/6.-System-Support-for-MIMDRAM.md` (score: 271.4), `paper_secs/.../Mess.../IV.-PERFORMANCE-CHARACTERIZATION-MEMORY-SIMULATORS.md` (score: 3097.2)

---

#### 方法: 自研 Event-Driven Multi-Chiplet GPU Simulator

以 calibrated macro-event 替代逐 cycle 模拟，牺牲微架构精度换取大规模可扩展性。Orders in Chaos 论文核心贡献之一。

- **核心机制**：
  - **Event Types**：COMPUTE_START/END (GEMM kernel 执行), MEM_READ/WRITE (HBM 访问, LLC hit/miss 判定), D2D_SEND/RECV (Die-to-Die 数据传输), ALLOC_START/END (Expert 分配决策)
  - **Resource Manager**：每个 die 维护 compute_queue, memory_queue, d2d_link_status; 全局 contention_tracker, congestion_detector
  - **校准验证**：8×H100 DGX 实测——(a) 单 GPU 执行一个 MoE expert (3 GEMMs, 不同 batch size); (b) 两 GPU P2P 数据传输 (4KB-4GB payload)。验证误差 <5%
  - **可处理规模**：25-die wafer-scale 系统 + batch 16,384 tokens + seq 1-131K
- **局限性**：无法捕获微架构级瞬时行为 (如 pipeline stall pattern)
- **开源情况**：开源 Apache-2.0 (GitHub: zhongkaiyu/waferscale_gpu_moe_sim), expert selection traces >150GB JSON on HuggingFace
- **评估速度**：秒-分钟级
- **来源**: Q6.5, Q6.6; vault: `paper_secs/secs_moe/Orders in Chaos.../A.-Experiment-Setup.md` (score: 324.6), `experiment_notes/硬件实验笔记/RPU.md` (score: 627.8)

---

#### 方法: Mess Simulator — 分析反馈型内存模拟

采用反馈控制理论的分析方法模拟内存系统，不建模内存微架构，依赖实测 BW-Latency 曲线作为 oracle。

- **核心机制**：
  - **PI 控制器循环**：每 1000-memory-op window: messBW_i = 当前估计带宽位置 → Latency_i = 从 BW-Latency 曲线读取 → CPU simulator 以 Latency_i 运行 window → cpuBW_i = 实际模拟到的带宽 → 若 |cpuBW_i - messBW_i| > threshold 则 messBW_{i+1} = messBW_i + convFactor × (cpuBW_i - messBW_i) 自适应收敛
  - **速度/精度**：gem5+Mess 误差仅 3% (传统 Ramulator 2 误差 52%), 速度比 Ramulator 快 13-15×
- **局限性**：无法揭示微架构瓶颈根因
- **适用范围**：当需要快速评估新内存技术 (如 CXL expander) 而缺乏详细微架构模型时
- **来源**: Q6.5; vault: `paper_secs/.../Mess.../IV.-PERFORMANCE-CHARACTERIZATION-MEMORY-SIMULATORS.md` (score: 3097.2)

---

#### 方法: RTL/EDA 工具链 (Synopsys/Cadence + Verilator)

从 RTL 设计到面积/功耗/时序报告的完整 EDA 流程。

- **核心机制**：
  - **RTL 设计**：SystemVerilog / VHDL / Chisel / BSV
  - **功能仿真**：Synopsys VCS / Cadence Xcelium / Verilator (开源, Verilog→C++ cycle-accurate 模型)
  - **逻辑综合**：Synopsys Design Compiler / Cadence Genus → 面积/功耗/时序报告 (target 特定工艺: TSMC 28nm/7nm/5nm/N2)
  - **HLS**：Catapult HLS + SystemC (RPU chiplet compute fabric)
  - **OpenPiton Metro-MPI**：Verilator 将每 tile RTL 转为 C++ model, 所有 tiles MPI 并行仿真, 但 64-core 单 datapoint >5 hours, 全曲线 >1 year
- **关键使用模式**：RTL 仿真仅用于功能验证 (验证正确性), 综合用于面积/功耗报告, cycle-level/event-driven simulator 用于系统级性能 DSE (因 RTL 仿真太慢无法运行完整 benchmark)
- **来源**: Q6.5, Q6.6; vault: `experiment_notes/硬件实验笔记/RPU.md` (score: 627.8), `experiment_notes/硬件实验笔记/Focus.md` (score: 394.3), `paper_secs/.../BitMoD/` (score: 323.3)

---

### 分类: 硬件验证平台与 Benchmark

#### 方法: FPGA 原型验证 — Xilinx VCU118 / Alveo U280 / ZCU104

将关键子系统综合到 FPGA，大规模部分使用 cycle-accurate simulator 通过 PCIe 与 FPGA 通信的混合仿真策略。

- **核心机制**：
  - **IANUS 验证架构**：FPGA (Xilinx VCU118) 上实现 PIM Control Unit + PIM Memory Controllers → FMC 连接器 → 商业 GDDR6-AiM PIM 芯片；NPU Simulator (因 NPU 过大无法放入单 FPGA) → PCIe → FPGA → PIM。验证指标为功能正确性 (GPT-2 Base/Medium/Large/XL on WikiText-2 perplexity vs full-precision)
  - **Tandem Processor**：Xilinx Alveo U280, 新兴算子加速器, latency/throughput vs GPU baseline
  - **FEATHER**：Xilinx ZCU104, PYNQ + Jupyter notebook, 可重构 dataflow accelerator, 端到端推理延迟 vs DPU/Gemmini
- **关键约束**：单 FPGA 逻辑容量有限，验证指标为功能正确性 (非性能——原型频率远低于 ASIC 目标频率)
- **来源**: Q6.6; vault: `paper_secs/.../30-IANUS/7.3-IANUS-System-Prototyping.md` (score: 1063.2)

---

#### 方法: MLPerf / MoE-CAP / PARSEC — Benchmark 套件

- **MLPerf Inference/Training**：行业标准 benchmark，芯片级设计论文广泛引用。SCAR 从 MLPerf curated 5 个 multi-model scenarios 代表数据中心多租户场景评估异构 MCM 调度器。指标包括 TTFT (Time-To-First-Token) 和 TPOT (Time-Per-Output-Token)
- **MoE-CAP**：MoE 专用 benchmark，三维评估框架 (Cost-Accuracy-Performance)。指出 MLPerf 在 MoE 系统的三个缺陷：(1) 缺乏 CAP 关系理解原则；(2) MBU/MFU 因忽略 expert 稀疏激活而高估 (batch>1 时高估 1.5×-3×)；(3) 不考虑异构资源综合成本。提出三维 cost model: C_hardware = (C_GPU + C_CPU) + (C_C2M + C_PCIe + C_NVLink) + (C_HBM + C_DRAM + C_SSD); C_token = (C_hardware + C_energy × $/kWh) / (T_token × R)
- **PARSEC**：学术标准多线程共享内存 benchmark，DFBM 全系统模拟使用
- **自定义 Expert Traces**：Orders in Chaos 从真实 MoE 模型 (DeepSeek V3, Kimi K2, Llama4) 收集的 expert selection traces (>150GB)，因为标准 benchmark 无法捕捉 expert selection skewness 和 token-level routing pattern
- **来源**: Q6.6; vault: `paper_secs/.../MoE-CAP/2-Background-and-Motivation.md` (score: 228.9), `paper_secs/.../SCAR/SCAR.md` (score: 2386.9), `paper_secs/.../DFBM/VII.-EVALUATION.md` (score: 3884.7)

---

## 方法间关系

### 替代关系

- **Chiplet (MCM-GPU) ←→ Wafer-Scale (WSE-2/SoW)**：
  - Chiplet 优势：灵活扩展 (逐步增加 chiplet)、异构工艺 (compute 3nm + IO 7nm)、成熟商业生态 (B200, MI300X)、KGD 良率策略可靠、供电/散热成熟、代际升级灵活
  - Wafer-Scale 优势：消除 off-chip 通信 (20 PB/s 片上 BW vs HBM ~2 TB/s)、巨量片上 SRAM (40GB vs ~50MB L2)、高并发 PE (850K vs 14,592)、适合极度 memory-bound 或 all-to-all 密集负载
  - 两者并非 mutually exclusive——TSMC SoW 本质上是 "wafer-scale chiplet" (晶圆级封装集成多个 chiplet 级 die)，模糊了传统边界
  - 商业趋势：NVIDIA 正从 2-chiplet (B200) → 4-chiplet (Rubin) 演进，chiplet 是近期主流；Wafer-scale 仅 Cerebras CS-2/3 和 Tesla Dojo 有限商用

- **Chiplet D2D 通信 ←→ Multi-GPU EP NCCL 通信**：
  - Chiplet EP: D2D XY routing over interposer (~200ns/hop)，瓶颈为 intra-package D2D bandwidth
  - Multi-GPU EP: NCCL AlltoAll over NVLink/NIC (~10-100μs 跨节点)，瓶颈为 inter-node network bandwidth
  - Chiplet 方案将通信延迟降低 2-3 个数量级，但引入 NUMA 效应 (local vs remote HBM 10-15×)

- **2D Mesh NoC ←→ Torus NoC**：
  - Mesh: 更简单的路由 (XY 天然 deadlock-free)、更低的硬件开销
  - Torus: 更高的路径多样性 (wrap-around 边)、更适合大规模 (>64 nodes) 和 all-to-all 密集通信
  - All-to-All 占 MoE 训练 41.5%-95.7% 时间 → Torus 的路径多样性在此场景具决定性优势 (HalfRing+DimRotation 2.28× vs mesh baseline)

- **PIM ←→ GPU (Batch Size 分水岭)**：
  - 低 batch (B<32): PIM 优势 (MoE expert FC ArI 低, PIM 4× GPU HBM BW)
  - 高 batch (B>64): GPU 逆转 (FC ArI 超过 PIM Ridge Point 8 Op/B, GPU RP 281 Op/B)
  - MLA + MoE 架构下大 batch 已成主流，PIM 仅适用于低 batch/低序列长度场景

- **DFBM ←→ MTR/DeFT/RC (Chiplet NoC 死锁避免)**：
  - MTR: 边界路由器转向限制，负载不均、非拓扑无关
  - DeFT: VC 隔离上下行流量，拓扑无关但每 VN 需 ≥2 VC
  - DFBM: 零侵入、拓扑无关、标准化、全开销约束于 interposer，saturation throughput +14% vs alternatives

### 互补关系

- **Two-Level CP + Multi-Die Task Allocation + ATU/PDU**：三者构成完整 wafer-scale GPU 优化栈——Global CP 运行 Task Allocation → Local CPs 并发 dispatch SMs → ATU/PDU 消除重复 D2D 数据搬移。叠加效果: Allo+Pred 7.0-7.5× throughput, hop count -213×, 总面积/功耗 <0.04%

- **PIM (物理近存计算) + ERAS (调度层 expert placement 优化)**：正交叠加——PIM 提供物理层高带宽近存计算能力，ERAS 提供调度层 residency-aware gating 减少远端 expert 选择，两者无冲突可同时使用

- **Chiplet + Active Interposer + DFBM**：Chiplet 提供计算粒度，Active Interposer 提供通信基础设施，DFBM 确保跨 chiplet 通信死锁自由

- **Duplex xPU+PIM 异构并发**：Attention (compute-bound) 在 GPU 执行 ∥ Expert FFN (memory-bound) 在 PIM 执行，算子级流水线重叠实现 ~2× 并发加速

- **Node-Link Balance (LP + Bayesian Optimization)**：Node Balance (LP) 平衡计算负载和通信量 + Link Balance (BO) 最小化链路级拥塞，协同优化 MoE all-to-all 的 NoC 通信效率

- **Preload Reordering (编译时) + Dynamic Pre-Broadcast (运行时)**：编译时优化 execution plan 消除结构性拥塞 + 运行时利用 expert activation 时间局部性预广播热门 expert

- **模拟器验证链**：RTL 仿真 (功能验证) → 逻辑综合 (面积/功耗报告) → Cycle-level/Event-driven simulator (系统级性能评估) → FPGA 原型 (关键子系统验证)，形成从设计到评估的完整验证链

### 依赖关系

- **Multi-Die Task Allocation → Two-Level Command Processor**：Task Allocation Algorithm 依赖 Global CP 运行 (A76-class ARM core 提供算力)
- **ATU/PDU → D2D Controller**：ATU/PDU 嵌入在每个 die 的 D2D controller 中，依赖 D2D Controller 的硬件实现
- **Duplex 调度 → Ridge Point 分析**：算子-设备匹配决策依赖正确的 Op/B 计算和 Ridge Point 硬件特征值
- **HD-MoE Link Balance → 2D Mesh NoC**：Link Balance via Bayesian Optimization 依赖 2D mesh NoC 的 XY routing 和 link schedule dictionary
- **DFBM → gem5+Garnet 验证**：DFBM 死锁避免正确性验证依赖 gem5+Garnet cycle-accurate NoC 模拟和 CDG 理论分析
- **3D NMP Hybrid Bonding → Logic Die 独立工艺**：3D NMP 的 Hybrid Bonding 技术使 Logic die 可独立于 DRAM 工艺节点 (如 7nm CMOS)，是 3D NMP 架构可行性的物理前提
- **PIM Bank-Level Parallelism → Power Delivery Network**：全 Bank 并发 MAC (MAC_ABK) 受限于 PDN 峰值供电能力和热密度，是 PIM 并发能力的物理约束

---

## 本层不确定性

1. **DiT (扩散模型) 芯片级设计证据严重不足**：所有六个答案中，DiT 在 chiplet/wafer-scale/PIM/NoC 上的芯片级设计均标注为"推断"或"该链条节点无 note evidence"。笔记中仅有 ELK paper 对 DiT-XL 做了推理性能评估 (Fig 23, compute-intensive 特性使互联优化收益较小)。DiT 的串行迭代依赖、全图 batch GEMM 特性与 LLM decode 的 memory-bound GEMV 有本质不同，现有芯片设计结论不能直接迁移。**未来需要 DiT-specific 的 chiplet/wafer-scale benchmarking 和 PIM 适用性分析。**

2. **多模态异构编码器协同的芯片级方案无 note evidence**：vault 中存在 MoDES、DiEP 等多模态 MoE serving 笔记，但均侧重算法/软件层 (expert skipping, multimodal gating)，不涉及 chiplet/wafer-scale 芯片设计。Chiplet 的异构集成能力在此场景最具潜力 (不同模态编码器 chiplet 使用不同工艺/架构优化)，但无实验验证。

3. **Video 时空注意力在 wafer-scale 上的实际 mapping 无 note evidence**：WSE-2 的 40GB SRAM 对于大规模 Video DiT 的长序列 KV cache 是否充足 (262K tokens × 32 heads × 128 dim × FP16 ≈ 2.1GB per layer)、weight streaming 对随机 KV 访问的效率等问题均待验证。笔记中 V-Rex 论文涉及 Video LLM 的 KV Cache streaming 但非芯片级硬件方案。

4. **ReRAM Crossbar 在 MoE/DiT/多模态/Video 场景下的定量性能数据缺失**：笔记中 ReRAM 相关信息主要出现在参考文献中，缺乏针对具体模型负载的 ReRAM 集成设计方法和实验结果。模拟计算精度-能效权衡缺乏定量证据。

5. **NPU 芯片级设计证据不足**：仅在 Meta MTIA 2i 芯片论文中有 NPU-like 加速器的 NoC 设计讨论。华为 Ascend、Google TPU 等 NPU 的芯片级多算子并发设计方法未在 vault 中找到充分证据。

6. **NoC 拓扑定量对比不完整**：笔记提供了 2D mesh (Dojo 5×5, SoW 8×3) 和 Torus (TPUv3/v4) 的详细数据，但对 Butterfly、Benes、Fat-Tree 等其他拓扑在 AI 推理负载下的定量对比仅 FEATHER paper 间接提及。Torus Networks 论文主要讨论多芯片集群级 (TPU Pod) All-to-All 优化，其 HalfRing/DimRotation 方法在单芯片内 NoC 的适用性需进一步验证。

7. **PIM 在大 batch 推理的适用性边界已明确 (笔记显示)**：大 batch (B>64) 下 GPU 优于 PIM (因 Ridge Point 差异)，与 MLA + MoE 架构主流趋势一致。但笔记未说明未来 PIM 计算能力提升 (如先进工艺 Logic die in 3D NMP) 是否能改变此边界。

8. **商用 PIM 软件栈成熟度低**：笔记显示商用 PIM (AiMX, HBM-PIM) 软件栈仍在早期阶段——主要依赖 vLLM integration 和学术模拟器 (Ramulator, PIMphony)，缺乏生产级编译器、调试工具和性能分析工具链。

9. **ASIC 测试芯片流片证据缺失**：多数芯片设计论文未到达 tape-out 阶段。BitMoD 和 RPU 使用 RTL 仿真 + 综合报告，IANUS 使用 FPGA 原型，但均未明确提及硅验证结果。唯一涉及商业 ASIC 的是 Cerebras WSE-2/3 和 Meta MTIA 2i 论文。

10. **Wafer-scale 商用成熟度有限**：TSMC SoW 仍在 roadmap 阶段 (2025 ECTC 展示 SoW-X)，Tesla Dojo 是已部署的 wafer-scale 系统代表但内部架构细节公开有限。Cerebras WSE-2 是当前唯一大规模商用的晶圆级 AI 芯片。

11. **供电网络 (PDN) 和热管理的芯片级定量对比缺失**：vault 笔记未涉及 PDN 设计细节 (IR drop 分析、decoupling capacitance 需求、TSV 供电密度等)。Wafer-scale 的供电和散热是重大工程挑战——WSE-2 46,225mm² 上的 IR drop 控制和热梯度管理对并发效率有直接影响 (热节流限制所有 die 同时以全 TDP 运行)，但缺乏笔记证据的定量分析。

12. **DAWNBench 未发现笔记证据**：在所有四个搜索目录中均未发现 DAWNBench 的具体使用案例。MLPerf 是芯片设计领域绝对主流的 benchmark 标准。DAWNBench 可能已在工业界被 MLPerf 取代。

13. **多模态和 Video MoE 场景的工具链标准化缺失**：笔记中所有自研工具链均针对特定模型 (Llama3, Qwen3, DeepSeek V3, Llava-Video)，尚无类似 MLPerf 的芯片设计 benchmark 标准覆盖多模态和 Video MoE 推理负载。

[HORIZON_SUMMARY_DONE] L6
