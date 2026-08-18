## PIM（Processing-in-Memory，存内计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM 把计算逻辑集成进存储器内部或近旁，让数据"就地"被处理而不必搬到处理器。动机是内存墙：处理器与内存的速度鸿沟持续扩大，LLM 推理（CompAir 引：OPT-66B 的 PCIe 传输占 90% 推理延迟）被外部带宽主导，而 PIM 可利用内存内部带宽——UPMEM 内部比外部高 6.7×、AiM 高 16×（CompAir 摘要称相对 GPU 5–20×）。按存储介质分 DRAM-PIM（SIMD 向量并行、容量大）、SRAM-PIM（亚 10ns 矩阵、容量小）、NVM/NAND-PIM；按计算位置分 PIM（存内）与 PNM（近存）。CHIME 补充 DIMM-PIM 家族：bank PU（DRAM 工艺、bank 级存内、>30× 主机带宽）+ rank PU（buffer chip 逻辑工艺、rank 级近存、约 4×），2TB+13.0TB/s 的均衡配置是 AFD 场景下同时缓解 HBM-PIM 容量瓶颈与 CPU 带宽瓶颈的关键（见 CHIME-PIM 条目）。CompAir 的核心判断：LLM 的 memory-bound 与 compute-bound 算子随 batch/序列长度动态并存，单一路线（纯 DRAM-PIM 或纯 SRAM-PIM）都无法高效覆盖，需要混合。COSM（ISCA'26，移动端）补充"共享内存空间 CPU/PIM 并发"场景：LPDDR5-6400（2ch×2rank、16 bank 级 PIM 单元/rank、1GHz、6.4 TFLOPS、6.4 GB/s，按三星 LPDDR5-PIM 流片芯片 [36] 建模）与 CPU 物理共享同一内存空间，OS 逻辑隔离替代 UPMEM 式静态分区；LLM（BLOOM-1B1/DeepSeek-R1-1.5B/Qwen2-0.5B，16-bit 输入/8-bit 权重）与后台应用并发执行，利用"CPU 偏外部带宽、PIM 偏内部带宽"的互补性收割空闲内部带宽——代价是 bank 冲突、命令总线拥塞与 CPU-mediated 传输干扰，需可抢占命令 + 空闲感知调度（见本库同名条目），实测 PIM 吞吐 +2.8×（较 Chopim）、CPU 降速 <2.0%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
CompAir 的 PIM 系统组织（device→channel→bank 三级）：32 台 PIM 设备经 CXL switch 互联；每设备 32 通道；每通道 16 bank；每 bank 由 DRAM-PIM（32MB、16 输入 BF16 MAC）+ 4×8KB SRAM-PIM 宏经 hybrid bonding 1:1 配对。执行例子（batch=64 decode）：投影/FFN 权重常驻 SRAM-PIM，每个 token 的激活经 DRAM→HB 进入宏完成矩阵乘，只有结果向量跨 bank 参与归约；QK^T/SV 的 GeMV 由 DRAM-PIM 就地完成。PIM 的短板是标量级灵活操作（RoPE 邻居交换、Softmax 的 exp）：传统做法是集中 NLU/CPU（PNM），CompAir 实测长上下文时非线性通信+计算可超 25% 总延迟，于是改用 NoC 在途计算去中心化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
商用/流片代表：UPMEM DDR4-PIM（DPU）、SK hynix AiM（1y-nm GDDR6、1 TFLOPS、ISSCC'22）、Samsung FIM-DRAM/HBM-PIM；SRAM-PIM 代表为 28nm 64kb 数字域浮点 CIM（31.6 TFLOPS/W、ISSCC'23）。仿真栈：ramulator2.0（DRAM）+ Booksim（NoC）+ CENT 模拟器（指令执行/CXL），CompAir 全套开源于 https://github.com/Man0xbfc00380/comp-air.git（arXiv:2509.13710）。使用要点：按算子形态分派硬件（GeMM→SRAM-PIM、GeMV→DRAM-PIM）；矩阵切分（output-split/input-split）决定 bank 间归约开销；非线性需要 NLU 或 NoC 在途计算等配套机制。COSM 的用法要点：单主机设计 + bank 级命令（PIM_Exec/PIM_Pause/传输命令对），内存控制器全知 DRAM 状态即可细粒度交错 CPU/PIM 命令；能量上 PIM 计算按三星流片芯片参数核算，DRAM 功耗用 DRAMPower，每 token 能耗较 AsyncDIMM-Bank/Chopim 降 1.34×/1.61×。DCC（ISCA'26）补充 near-bank PIM 的共性抽象与数据布局约束：把 PIM 组织抽象为 System（Host xPU + Host 内存空间 + PIM 设备）/PIM Group（同 channel/rank 的多个 core，支持 group 级广播指令）/PIM Core（单处理单元 + 本地 bank）三级；并指出 Host 与 PIM core 对数据布局的要求相反——Host 需要连续元素跨 bank 分布以利用 bank 级并行（cache line 粒度访问），PIM core 需要连续元素落在自己本地 bank 内以最大化本地带宽，因此 PIM kernel 执行必须包含三步：①输入数据重排（连续元素搬入同 bank）→②PIM core 计算→③输出数据重排（合并部分结果或转回 Host 布局），且重排通常经 Host 内存总线完成、开销可占主导（TVM 式编译方案中达 kernel 时间的 64.68%）。Web 证据（Hot Chips 33/IEEE Micro 2022）与论文一致：Aquabolt-XL 每 die 32 处理器、每处理器 2 个 16-bit FP16 乘法器 + 2 个加法器（16-lane SIMD）、全栈 128 处理器 1.2 TFLOPS。

HybridSpec 补充视角（ISCA'26，HB vs PIM 的异构平台对比）：两者都追求"把内存受限计算靠近高带宽内存"，但制造与映射方式不同——(1) 制造：PIM 把逻辑嵌进 DRAM 工艺（驱动电流受限、单元刻蚀高电容，不适合高效逻辑，常为简单 MAC 牺牲 ~50% 容量）；HB 在独立工艺分别造逻辑与 DRAM 再键合，fine-pitch（2-3µm）键合 via 占 DRAM die 面积 <3%，容量-带宽权衡更优。(2) 映射：PIM 只能集成简单 MAC，非线性算子和数据移动需配套 XPU，强制算子级（operator-level）划分（每次 forward 跨设备搬运逐层激活）；HybridSpec 的 HB 栈集成完整逻辑 die，支持模型级（model-level）划分——draft 模型整体放 HB 栈、target 放 XPU，通信只在 draft-verification 边界（图 22 对比数据移动量）。结论：HB 比 PIM 更适合"整模型级"异构部署与低通信开销。

从硬件架构角度拆解（对比例）：PIM 系统 = XPU（处理非线性/搬运）+ PIM 栈（MAC 阵列就地算 GeMV），同层算子被切到两处、每前向都要搬运激活与部分和；HybridSpec = XPU（target 全模型）+ HB 栈（draft 全模型），模型边界即通信边界，只有 token 列表与 draft KV 跨设备。

实现与使用：PIM 的代表是 HBM-PIM/AiM/UPMEM 等；HybridSpec 用它作"为什么不用 PIM"的对照——对 SD 这类"整模型极化"的场景，模型级映射 + 独立工艺逻辑 die 的 HB 栈是更优选择，PIM 的算子级映射仍适合注意力卸载类工作（NeuPIM/IANUS/SpecPIM）。

Taking Analytic Databases to the Bank 补充数据库领域视角（ISCA'26，BLIMP OLAP）：PIM 用于 OLAP 分析型数据库时，重点是"端到端"而非单算子——(1) BLIMP 型 PIM（DDR bank 内 RISC-V 核）以 memory mode / compute mode 双模式工作：memory mode 下 DIMM 当普通内存，compute mode 下各 bank 核接管本地 bank 执行指令，期间 host 无法访问该 bank 数据；bank 间无直连，跨 bank 数据交换必须由 host 读-写中转。(2) PIM 的收益来自 bank 级并行与 bank 内带宽（论文配置 512 core、约 12.5TBps bank 内带宽），但受两大约束：核性能弱（200MHz RISC-V）与 bank 容量有限（32MB/bank）——传统 hash join 的"用更多空间换低延迟"设计无法直接映射到 PIM（只能使用有限的预分配 scratch 内存）。(3) 端到端评估表明：隔离算子外推（多数 PIM 数据库研究的做法）会忽略 relayout、物化、算子链格式兼容等系统级开销，导致平均 22% 查询时间花在 relayout、比全栈协同规划慢 3.2×；PIM 感知规划（晚物化 + 低选择性优先 join 序）比 CPU 启发式快平均 28%（最大 40%）。SSB SF100 上 BLIMP-S/BLIMP-V 相对手调 C++ 基线 1.4×/2.3×、相对 DuckDB 3.1×/5.8×。

涉及论文标题：
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- Taking Analytic Databases to the Bank

MERIDIAN 补充视角（ISCA'26，CXL Type-3 LPDDR5X-PIM 的去中心化 RAG）：MERIDIAN 是"PIM 当计算设备 + CXL 当互联"的去中心化形态——32 个 CXL Type-3 PIM 设备（16 DAC + 16 CEC）各 512 GB/32 TFLOPS 经 CXL switch 互联，共 16 TB 容纳 TB 级文档 KV 库；PIM 设备 = CXL 控制器 + 8 个 LPDDR-PIM package（8-channel PIM 控制器、128-bit 总宽、每 channel 4 个 16-bit die），PU 置于 bank 旁（16 FP16 比较/乘法/加法器 + 4KB buffer，All-Bank-Mode），NMU/softmax 单元/8 个 BOOMv2 核做归约与 softmax。对"PIM 能力边界"的再定义：不追求 SRAM-PIM 式亚 10ns 矩阵，而是用 LPDDR5X 大容量 + bank 级 GEMV/skinny-GEMM + 内存侧 LUT 非线性 + 专用 softmax 硬件覆盖 RAG 全部 memory-bound 算子，从而无需 NPU/GPU 外部引擎；异构处理器（DAC/CEC 微架构同构）允许动态负载迁移。文档注意力分解让 PIM 就地算文档注意力，只交换紧凑统计量，通信占比 ≤6.34%。
