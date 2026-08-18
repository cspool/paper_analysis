## Roofline 模型（Roofline Model）

术语解释
性能分析模型：`P = min(π, β×I)`，把可达到性能表达为峰值算力 π 与带宽×算术强度 β×I 的上包络（roof），用于判断负载是 compute-bound 还是 memory-bound。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
横轴算术强度 I = FLOPs/字节访存量，纵轴性能（如 GFLOP/s）。斜线 β×I 是带宽顶、水平线 π 是算力顶，交点（ridge point，I=π/β）左侧为 memory-bound 区、右侧为 compute-bound 区。ATX 论文用它做 SDDMM 的能力分析（图 16）：(a) 纯核执行时所有点落在 compute-bound 区（核算不过来，即便换成 oracle 预取器、消除全部访存停顿，也够不到核的 roofline——受依赖、前端停顿等 core-bound 因素限制）；(b) 加 ATX NCA 后 ridge point 大幅右移、所有矩阵落入 memory-bound 区（算力不再是瓶颈，带宽成为限制），此时性能到 roofline 的差距来自 UTE 资源有限与预取距离启发式次优；Inf UTE + 每矩阵静态最优预取距离可达 roofline（LIV 除外，其差距源于 stride 式任务预测对不规则访存失准）。注意论文把纵轴改为 Giga Vector Ops/s 而非 FLOP/s，因为稀疏 kernel 的许多向量操作不产生 FLOP。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SDDMM 分析实例：某矩阵落在图 16(a) 核 roofline 的 compute-bound 区 → 挂 NCA 后它相对"带宽×I"线的位置决定了新瓶颈：若仍低于斜线说明 UTE 供数或 NCA 利用率不足，若贴到斜线说明纯带宽受限——论文用三个配置（默认 UTE、Inf UTE、Inf UTE+Best PF Dist）逐步逼近斜线来定位剩余性能缺口的具体来源（UTE 资源 vs 预取策略）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
标准用法：测峰值 π（厂商标称或 microbenchmark）与实测带宽 β，算应用 I，定位所在区后按区优化——memory-bound 提复用/换精度/做融合（提高 I），compute-bound 减 FLOP 或换低精度。论文的变体用法：把"纵轴单位"换成更有意义的领域指标（vector ops/s），把"oracle 预取器曲线"作为先前预取工作的理论上限参照，把"逐矩阵静态最优预取距离"作为任务预取机制的 oracle。局限：roofline 只给理论上限与瓶颈类别，不给具体优化手段。CHIME 在 AFD 场景的扩展 = DRM（Disaggregated Roofline Model）：把单一设备的 roofline 扩展为两条 "batch—token 吞吐" 曲线（GPU 的 Line-FC 与加速器的 Line-ATTN）+ 容量线（Line-Cap），系统吞吐取两曲线较低者，归纳出"吞吐由加速器带宽/容量较弱一维决定"（Liebig's Law）；曲线可理论推导也可 profiling 实测。

HybridSpec 补充视角（ISCA'26，roofline 作为运行时投机调度的利用率判据）：论文用 P=min(π, β×I) 判断系统瓶颈——硬件利用率取决于"算子算术强度 I（=每前向处理的 token 数 × 每 token 计算/访存比）"与"硬件计算-带宽比"的匹配：I 低于比值 → memory-bound（计算闲置），高于 → compute-bound。GPU 计算-带宽比 100-300，需数百 token/前向才能算满（Fig.4 实测吞吐随 token 数上升后饱和）。SD 让 draft（低 I、memory-bound）与 target（高 I）极化，故异构分配：HB 栈（4TB/s、计算弱）接 draft，XPU（算力强、带宽 1.1TB/s）接 target。调度层用每轮实际 I（token 数/迭代）对比各单元 roofline 拐点：HB 栈 I < roofline 时增大 tree width（多探索）、XPU I < roofline 时加倍 draft budget，把任务算术强度推向拐点以压满带宽/算力（Algorithm 1，详见系统架构层 Utilization-aware Speculation）。

从硬件架构角度拆解（HybridSpec 例）：draft decode 每轮 1-2 token/请求、算术强度低，落在 HB 栈 roofline 的 memory-bound 区 → 提高 tree width 提升每轮 token 数（更贴近带宽顶）；target verification 一次验证多个 draft token、I 高，落在 XPU roofline 的 compute-bound 区 → 提高 budget 摊薄权重带宽。二者把"固定硬件、动态负载"的利用率缺口（FIFO 55.63% → PFS 62.43% → CHK 66.17% 平均利用率递增）通过运行时调 I 弥合。

实现与使用：用 SVR 对 (draft budget → 最优 tree width) 拟合查找表 T（拐点 p），运行时按 roofline 比较调参（预算上限 B=32 防拒绝 token 浪费）；是 roofline 从"离线设计空间分析"扩展为"在线调度目标"的用法——与 CHIME 的 DRM（分离式 roofline）同属 roofline 的系统化扩展。

MLX 补充视角（ISCA'26，roofline 定位 GPU 上蝴蝶稀疏 kernel 的性能缺口）：对 FFT 与 BSMM kernel 做 roofline 分析——它们相对稠密 GEMM（TensorCore，OI 高）运算强度（OI，有效 FLOPs/DRAM 字节）低，落在带宽受限区，但实测仍远低于 CUDA 带宽 roofline（H100 上用优化 cuFFT 的 Llama2-7B prefill N=512/8K，Fig.3），说明缺口不止 memory-bound：多级 strided/shuffle 访问破坏局部性、高 cache miss（Fig.2 AGX Orin profiling）+ stage-wise 依赖与 GPU 批量同步/tile 规则执行错配（执行单元不匹配，蝴蝶 kernel 只能跑 CUDA core 而非 TensorCore）。MLX 用该分析驱动架构设计：把跨层依赖转成片上 skip-hop 路由后，同一 butterfly workload 在 MLX 上 roofline 利用率达 52%-84%（蝴蝶算子）/43%-75%（SWA），对比 AGX Orin 12%-29%/10.8%-31%、RTX-3090 8.2%-31%/8.9%-28%——roofline 从"GPU kernel 性能归因工具"延伸为"空间数据流架构的设计判据"。

NASZIP 补充视角（ISCA'26，roofline 论证 ANNS 的 memory-bound 本质并选择 DIMM-NDP）：论文用 roofline（Fig.3）对 SIFT/GIST/GloVe 上 HNSW（CPU）与 CAGRA（GPU）做分析——距离计算算术强度极低，两种平台都落在带宽受限区（memory-bound），因此聚合内部内存带宽是唯一有效杠杆；同时比较 SRAM-based PIM（高速片上 SRAM 算力强但容量小、每 bit 成本高）与 DIMM-based NDP（大 DRAM 容量 + 高内部带宽）在 roofline 下的取舍，最终选择 DIMM-based NDP。roofline 在此把"用什么硬件"变成定量判断：负载 memory-bound → 优先加带宽而非算力。

3DGS 加速器补充视角（ISCA'26，roofline 定位光栅化与 MLP 推理的算力/带宽瓶颈差异并设计 interleaved pipeline）：论文对统一可重构 PE 阵列的两个阶段做 roofline 分析（Fig.10 左）——rasterization 每个投影 Gaussian 9 参数、做 256×6 MAC（算术强度高，compute-bound）；MLP 推理 1 深度参数仅 6 MAC，两者运算强度相差约 30 倍，在 96KB 片上缓冲 + 38.4GB/s DDR5 的配置下 MLP 推理落在 memory-bound 区（带宽顶下方），因此 naive 的先全量 MLP 后光栅化流水 PE 利用率极低。roofline 在此指导出"按 subtile 做 fine-grained interleaved pipeline"：把 tile 细分为 subtile，把当前 subtile 的光栅化与下一 subtile 的深度装载/MLP 推理重叠，把 memory-bound 的 MLP 访存藏进 compute-bound 光栅化的计算时间里，使 PE 阵列两阶段都贴近各自 roofline。同时该分析也解释了为何 GPU 上 MLP-OIT 更慢（cuBLAS GEMM 低算术强度受带宽限制，1.59× 慢于 Radix sort）——roofline 从"架构设计判据"用于"跨平台（GPU vs 专用加速器）迁移性论证"。


PIPEWEAVE 补充视角（ISCA'26，把 Roofline 扩展为 multi-dimensional pipeline roof 作为 ML 性能预测的特征）：论文不满足于经典 Roofline 的单 compute roof + 单 memory roof，而是为每类异构指令 pipeline（Tensor/FMA/XU math pipeline 与 MIO 内存 pipeline）分别计算独立的理论性能上限（"roof"）：先用 Kernel Decomposer+Scheduling Simulator 得到真实 task 分布，再按 `C_p = N_ops,p / Th_p` 与 `C_mem = B / BW_mem` 求各 pipeline 的 demand 与理论周期，把执行效率（理论时间/实测延迟）按 pipeline 解耦绘制成独立饱和曲线（Fig.3，FlashAttention-2 在 A100 上 demand 增大时两条配置分别逼近各自 roof 并平台化）。经典 roofline 的 H800 vs H20 反例被用来证明单一 roof 的不足：H20 保留约 120% 的 H800 内存带宽但算力仅 ~15%，compute-to-memory 比极低→GEMM 极易把计算单元喂满，Roofline 估计贴近实测（MAPE 11%）；H800 算力巨大、实际难达峰值，Roofline 严重高估（MAPE 127%）。PIPEWEAVE 把每条 pipeline 的 demand/理论周期作为原始特征喂给 MLP，让模型自动学习"该 GPU 上 pipeline 并发与争用造成的实际低效"，从而兼得 roofline 的可解释边界与数据驱动的硬件专属修正——这是 roofline 从"瓶颈分类器"扩展为"特征生成器+ML 预测输入"的用法。
Rearchitecting-the-Datacenter-Lifecycle 补充视角（ISCA'26，把 roofline 从"单 kernel 性能分析"扩展为"数据中心生命周期 TCO 的吞吐上界引擎"）：论文在 AI Lifecycle Compass 框架中用 roofline 模型预测 LLM 推理的 TTFT 与 TBT——从 LLM 架构与参数量解析推导算术强度与内存占用，结合硬件峰值 FLOPS、内存带宽/容量、互连带宽/延迟与 TDP，对每个 (模型, GPU 代际) 组合计算延迟；随后在 SLO（TTFT≤400ms、TBT≤100ms）约束下不断增压负载直到违反 SLO，得到 goodput（最大可持续 RPS）作为容量/利用率点。roofline 在这里的独特用法：技术路线（FLOPS/带宽/TDP 线性外推）直接平移 roofline 曲面，把微架构变化传播为生命周期 TCO——如 Llama3-70B 在 A100 上 goodput/Watt 比 H100 低约 3×10^10、而 1B 模型 A100 仅低 18%，A100 对小模型 per-dollar 反而优于 H100（8–23%），从而决定"跳过 B100/B200、延长 H100/H200 寿命"的刷新策略。论文用 vLLM 在真实 GPU（T4/V100/A100/H100/H200）上验证 roofline 对 Llama3 的误差 <5%。这是 roofline 从"离线 kernel 归因"扩展为"跨 15 年容量规划与硬件采购决策"的分析模型。
RoCC 补充视角（ISCA'26，ROP 的 CC 计算容量 roofline）：RoCC 论文用反向工程结果做 ROP 侧 roofline 分析，证明 ROP 计算容量足够做 CC：V100 按 64 ROP @1GHz、每 3 cycle 4 个 32-bit（或 8 个 16-bit）操作估算峰值 85.33 GFLOPS（FP32）/170.67 GFLOPS（FP16）；A100 按 160 ROP、每 cycle 1 操作估算 640 GFLOPS（FP32）/1280 GFLOPS（FP16）。ring AllReduce 的操作强度 ≈0.0625 FLOPS/Byte（FP32，一次归约对应四次数据共享），远低于网络 roof（NVLink 300/600GBps），故 CC 是网络带宽 bound 而非 ROP 计算 bound——结论是 ROP 的算力足够支撑 CC，瓶颈始终在互连，这解释了 ROP 数量减半性能仅降 3.7% 的敏感性。

- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，roofline ridge point 作为 chunk memory/compute 标签）：RESONATOR 的 Performance Atlas 用 roofline ridge point I*=PeakFLOPs/PeakHBM_BW 定义 chunk 的算力强度阈值——chunk 算术强度 I（FLOPs/byte）满足 I<η·I* 判为 memory-bound（decode-heavy），否则 compute-bound（prefill-heavy），η 为可配置强度阈值；该标签直接驱动 Intra-GPU Sharing Engine 的模式选择（memory-bound→SM 分区互补场景、compute-bound→kernel 级共享 contending 场景）。同时用 roofline 思想做 stage 级资源画像（Table I：encoder 高 SM/低 HBM、prefill/decode 高 HBM/中 SM）与 kernel 级 profile（每 kernel 标 comp/mem + SM/HBM 用量），指导 wide/narrow 流配额。logical sharding 微基准以 A100 FP16 峰值 312 TFLOPS 归一化 MFU 验证 strided GEMM 布局代价。
涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- ATX: Accelerator Task Extensions
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
- MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
- Rearchitecting the Datacenter Lifecycle for AI
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

> **操作强度（Operation Intensity, OI）作为静态调度目标（源自 Harmonia）**：Harmonia 的静态分析层把 OI 直接作为候选块形状的选取目标：OI = OPs / Bytes_loaded（SRAM 驻留块上的有效操作数 ÷ 离片传输字节数），最大化 OI 即最大化复用、最小化离片流量。对候选 (T_M,T_K,T_N)，在全局密度 ρA/ρB 的独立近似下估计 E[nnz_A]=ρA·T_M·T_K、E[nnz_B]=ρB·T_K·T_N、有用 OPs ∝ ρA·ρB·T_M·T_K·T_N，并用可行性约束 s_val(E[nnz_A]+E[nnz_B]) + s_psum·T_M·T_N ≤ β·S_SRAM（β=0.8）过滤后，按估得的 OI 选最大者（Algorithm 1 离线枚举）。例：ρA=ρB=0.1 的 16×16 PE 阵列上，(64,128,64) 对 InP/Row/OutP 都得到稳健 OI 且满足约束。局限与 roofline 模型同源：OI 基于全局平均密度，tile 级稀疏不均匀时估计偏差大（论文实测 SRAM 流量高估 3.7×/3.1×），因此 Harmonia 只用它定安全起点，由在线 Profiling 与动态 Tuning 修正。
