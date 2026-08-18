## Accel-Sim / BookSim2 联合仿真平台（多 GPU 逐周期模拟）

术语解释
本论文的评估基础设施：Accel-Sim（GPU 微架构模拟器）+ BookSim2（cycle-accurate NoC 模拟器）集成，逐周期模拟 NVL32 上 MoE 训练/推理的执行与通信。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Accel-Sim（ISCA 2020，https://github.com/accel-sim/accel-sim）：trace 驱动的 GPU 模拟框架，对 SM 流水线、cache、张量核做逐周期建模；本论文用支持基础 Hopper 特性的最新版并扩展高性能 FP8 GEMM kernel 模型。BookSim2（ISPASS 2013，https://github.com/booksim/booksim2）：参数化 NoC 模拟器（拓扑、路由、VC、仲裁、flit 级时序），本论文用它建模 NVSwitch/NVLink 网络。二者集成后支持 32 GPU 并发执行 + 交换机网络互联，实现整系统 cycle-accurate 仿真。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
模拟流程：SASS 指令流（含 dymultimem.st / dymultimem.ld_reduce 与 target list）→ Accel-Sim 逐周期执行 SM/LSU/MultimemQ/MMU/AL TLB/token tracker → 通信请求进入 BookSim2 的 NVLink-NVSwitch 网络模型（900 GB/s、250ns、16B flit、VC/归约缓冲参数化）→ 交换机多播/归约后响应回到 GPU 内存 → 逐周期统计换算执行时间与流量。校准：对照 DGX-H100 实测，GEMM 与 DeepEP 通信算子在多种 shape/volume 下平均误差 <6%。本论文新增模型：FP8 kernel、多 GPU 并发、switch 网络集成、dymultimem 指令流水线、AL Table/AL TLB 与 token tracker 三表。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
二者均为开源学术模拟器，广泛用于 GPU 微架构与互连研究。使用方式：注入 workload（SASS trace / 通信算子）+ 硬件配置（SM 数、频率、网络参数），输出周期数/流量等指标。局限：模拟速度远低于真实执行，通常评估 MoE 层或端到端单步而非完整训练；本论文端到端训练以 16-way pipeline parallelism（16 NVL32 节点）+ 注意力数据并行建模。RTL 综合另用 Synopsys Design Compiler + TSMC 12nm 库验证硬件开销（与模拟器正交）。

Cassandra 补充视角（ISCA'26）：在 Accel-Sim 中实现 Cassandra encoder/decoder 以评估 GPU 集成性能：配置近似 Nvidia RTX 4090 与 Nvidia Jetson AGX Orin；因 Accel-Sim 不支持 Ada-Lovelace，RTX 4090 用 Ampere 架构 trace + 近似配置回放。模拟数据通路：权重/KV 压缩数据经 superblock 载入 L2 → Cassandra decoder 解压（unary/MX 指数解码 + bitmap de-sparsification）→ SM 执行标准 FP GEMM；每 token 周期结果结合实测接受率（0.74–0.91）换算吞吐（1.78–2.41× vs BF16）。Cassandra 的修改版模拟器未开源（论文未给链接）。

ConBin 补充视角（ISCA'26）：组合换为 ScaleSim + BookSim2 做晶圆级芯片仿真：ScaleSim（https://github.com/ARM-software/SCALE-Sim，32×32 阵列、8MB SRAM、Output Stationary 数据流）模拟核内 GEMM/attention 计算周期，BookSim2（8 VC、buffer 8、flit 32b、packet 16 flits、router 4 级流水、每 mesh hop 1 cycle）模拟故障修复后不规则 mesh 上的 NoC 通信——跨 die 边界直连与 die 内同延迟、多跳冗余 R-R 链延迟按 hop 数成比例。修改：拓扑注入故障核/链、冗余互连与 turn-prohibition 路由。Fig.9 用 BookSim2 隔离 hop 数与任务间 contention 对延迟的影响（hop 延迟 5–6 跳后饱和、任务间 contention 近线性增长——contention 是故障下主要延迟源）。输出每 chip 端到端 latency → 汇总 512 实例性能分布做分 bin 分析（premium-bin yield、SECC）；面积另用 McPAT。ConBin 仿真代码未开源（论文未给链接）。

DICE 补充视角（ISCA'26）：DICE（CGRA 替换 SIMD 后端的 GPGPU 架构）扩展 Accel-sim 的 GPGPU-Sim 性能模型做周期精确评估——新增 DICE 机模型（CC/CP 层级、p-graph 粒度 e-block 的 CS/FDR/DE/RE 四阶段、CGRA 位流装载与双缓冲切换、逐线程 II=1 流水 dispatch、LDST Unit/BRT 非阻塞 retire、TMCU 时序合并），并用 AccelWattch 扩展构建 DICE 动态功耗模型：共享组件（L1/共享内存/计算）沿用 per-access 能量，DICE 新增 SRAM（CGRA 开关、配置存储等）用 CACTI 建模，控制逻辑用 Cadence Joules 对 FreePDK45 综合的 RTL 做活动 trace 分析；结果折算 12nm 与 RTX2060S 对比。使用：CUDA 11.7 编译 Rodinia kernel → PTX → DICE 编译器产 p-graph 位流 → 仿真输出周期/IPC/RF 访问/功耗。artifact 开源：Zenodo https://doi.org/10.5281/zenodo.19278715 、GitHub https://github.com/jiayi-wang98/DICE-test-collection 、Docker 镜像 jiayiwang0710/dice-isca-eval:cuda11.7。

MXFFP 补充视角（ISCA'26）：用配置派生自 NVIDIA RTX 5090 GPU 的 Accel-Sim 评估 MXFFP Tensor Core：CUTLASS 生成 GEMM kernel 并提取指令 trace 输入模拟器，额外建模 MXFP 的 shared exponent 与 MXFFP 的 configuration bit 元数据访存流量，评估 GEMM 延迟（矩阵 256/512/1024）与端到端 LLM prefill 推理延迟（1024 token、batch 1/2/4/8，7 个 LLM）相对 BF16 的加速比。结果：MXFFP 与 MXFP 加速比几乎相同（4-bit 1024³ GEMM 2.7×、8/6-bit 1.8×；prefill 端到端 4-bit 2.08×、8/6-bit 1.55×、batch 1-8 时 4-bit 1.86-2.18×），证明 1-bit 配置元数据无延迟开销；能耗另用 AccelWattch 评估（见 AccelWattch 条目）。

MoE-Hub 补充视角（ISCA'26）：扩展 Accel-Sim + 修改 BookSim2 构建多 GPU cycle-accurate 模拟器（遵循 prior work [28][51] 方法论），baseline 配置为 8 GPU 经 4 个 NVSwitch 连接，模拟 NVIDIA DGX-H800 架构与拓扑。BookSim2 复刻 NVLink 设计：全双工链路、16B flit、单 flit 头、full-to-full 路由与 switch 级转发；每 GPU 400 GB/s、GPU↔switch 单向 250ns（往返约 1µs），带宽与延迟均按真实硬件校准，0.5MB-256MB 消息下模拟 All-to-All 时间与物理系统平均误差 4.36%。修改内容：(1) Accel-Sim GPU 模型中扩展 hub：AAU（RAT 16-bank 双口 SRAM + APT CAM、FIFO 驱逐 spill 到设备内存）、RPM（per-destination 全相联 SRAM 缓冲池 + Packet Scheduler：congestion-aware round-robin + consumer-aware 最小 RowID 优先 + timer bypass）、DAM（Dependency Table CAM + TB Status Counter + Global Counter/AllReady）；(2) GPU pipeline 加 st.rowsp 解码/路由轻量逻辑（共享存储 datapath、TLB 指令 flag 门控解析 MallocID→GPU ID）；(3) BookSim2 模拟 NVLink 互连参数。专家 GEMM 用 CUTLASS kernel；软件基线（Megatron-TE/FasterMoE/Tutel/Comet/CCFuser）在 Megatron-LM 上实现；端到端开销从真实 H800 运行时分解外推。工作负载：Mixtral 8x7B、Qwen2-MoE-2.7B、Phi-3.5-MoE（见 MoE 条目），序列 128-32768、token 分布 std 0-0.05、2-16 GPU。输出：MoE 层/端到端延迟加速比（1.40×-3.08× 层、1.21×-1.98× 端到端、96.8% 理想层）、消融（MH-Base/MH-PKT/MH-DEP/MH）、FLOPS scaling；硬件开销另用 TSMC 7nm 综合评估（0.49 mm²）。
RangeGuard 补充视角（ISCA'26，ECC 读延迟 +1 cycle 的 GPU 系统性能评估）：RangeGuard 用 Accel-Sim 配置为 NVIDIA V100 模型评估保护开销，因内存错误以天/月级发生，只在无错场景测性能。修改：假设 RangeGuard 需 2 cycle 返回无错数据（检错阶段后立即返回）、baseline 已有 1 cycle 检错，净增 1 cycle 读延迟；写延迟不加（额外编码 cycle 可被内存控制器队列等待或 DRAM 写命令到写数据到达的间隔隐藏）。评估 11 个 Rodinia [52] + Parboil [53] workload：geomean IPC 仅降 0.008%，最访存密集 workload <0.05%，计算密集几乎无影响——GPU 吞吐导向设计使 ECC 解码延迟可忽略。与 Cerberus 同套评估三角（错误注入定可靠性、Accel-Sim 定性能、DC 综合定面积/功耗），但 RangeGuard 的面积/功耗用 Synopsys Design Compiler + UMC 28nm @1GHz 综合（非 ASAP7/12nm），encoder+decoder 合计 11,100 μm²（RG 8b SSC，21,900 NAND2 ≈ 5e-7 Blackwell die）。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
- DICE: Enabling Efficient General-Purpose SIMT Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
