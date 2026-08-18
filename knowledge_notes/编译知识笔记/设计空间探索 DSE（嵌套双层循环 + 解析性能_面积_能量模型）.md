## 设计空间探索 DSE（嵌套双层循环 + 解析性能/面积/能量模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AutoFHE 的 DSE 联合优化设计空间 P = {CPE_micro, K, Sched}：CPE_micro 为 CPE 模板微架构参数（PoV/PoD/BFU/IBFU/PoE/PoC/PoK），K 为各模板类型的物理 CPE 数，Sched 为调度策略。采用双层嵌套优化：外层枚举硬件配置 (CPE_micro, K)（查面积模型、超预算跳过），内层对每个配置跑 GA 调度搜索求最优性能，全局最优随迭代更新（论文 Algorithm 2）。这是加速器 DSE 经典的 two-loop 模式（外层硬件点、内层映射/调度，如 Timeloop/ZigZag，DOSA 综述归纳），本论文的调度层在内层。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 评估模型：解析性能模型——总延迟 = max(计算延迟, 访存延迟)（FHE 两大瓶颈）；面积模型——Design Compiler 预表征原语（MAC、butterfly 等）成本按所选参数线性组合；能量模型——off-chip 访存 + on-chip 访存 + 计算单元 + 片上通信四源聚合（EIE [39] 风格）。目标函数可切 Latency 或 EDP（Latency x Energy）。
- 流程（Algorithm 2）：
```
DSE(G, AreaConstraint, Templates):
    Best = inf
    for Config = (K, CPE_micro) in GenCandidates(Templates):
        Area = AreaModel(Config)
        if Area > AreaConstraint: continue
        Schedule = SchedulingSearch(G, K)        # 内层 GA
        Metric = EvalModel(Schedule, Config, G)  # 解析模型
        UpdateBest(Metric, Config, Schedule)
    return Best
```
- 产出与案例：全局最优 (硬件参数, 调度) 对；Strix 同约束下自动发现 16 CPE + unrolling r=2；EDP 目标下自动选 24 bootstrapping 单元（延迟最优为 16）；300/640 GB/s 两种带宽下均命中 Pareto 最优点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- DSE 离线执行一次（eNPU 197.4 s、ALU1 329 s、ALU2 660 s），对比手工"数月"微架构设计+穷举；Python 实现。可扩展性：把 unrolling 因子 r 纳入设计空间后自动发现场景相关最优值（r=2 vs r=3），解决专家配置无法跨带宽/面积约束迁移的问题——这正是 DSE 相对手工调参的核心价值。

FlexQ-NDP 的编译空间 DSE 视角（ISCA'26）：DSE 用于 NDP 编译空间而非硬件设计空间——设计空间是编译策略的笛卡尔积：算子划分 {Part^M,N,K 于 Channel/Rank/Device/Bank 层级} × 缓冲四元组 (Val_Buf, Scale_A_Buf, Scale_W_Buf, Dequant_Buf) × 循环置换 {Order1=M→KOuter→N→KInner, Order2=N→KOuter→M→KInner} × DRAM 映射列数 Col_S。流程三步：(1) 编码设计空间；(2) 裁剪——padding 超 50% 的划分策略剪掉、DRAM 列利用率低于阈值的 tiling/映射组合剪掉，空间缩 4~5×；(3) 两阶段解析代价模型代替逐策略 cycle 仿真——先按策略统计缓冲 miss、dequant 次数、行切换次数（区分自然行切换与 buffer-miss 引发行切换并做重合抵扣），再用 Lat = t_CCDL·Num_col + Lat_RowChange·Num_row + Lat_Dequant·(1−Ratio_Overlap) 估延迟。效果：单个 batched MVM 的 DSE 从 3 小时降到 1 分钟、LLaMA2-7B 全模型 10 分钟；最优性（相对暴力搜索）W-A 平均 95.97%、W-Only 99.66%。与 AutoFHE 的差异：FlexQ-NDP 的 DSE 目标是编译策略（软件映射）而非硬件参数，代价模型以 DRAM 行切换事件为关键项——印证 NDP 性能由数据布局主导。

CODO 的调度 DSE 视角（ISCA'26）：把 DSE 用于 HLS 数据流的循环并行策略空间——设计变量 = 每个循环的 tiling/unroll/pipeline/array partition 组合，而非硬件参数或 NDP 编译策略。三阶段：(1) PA 初始并行度分配——profiling-based 性能模型（基本操作延迟/资源离线 profiling 作参数）按 trip count 与并行策略估计每循环 latency，按延迟比例分配并行度（最小 1）、保持比例整体放大到用户上限或资源极限，并优先 tile 与 FIFO 无关的循环维度（reuse buffer pass 的并行化合法性分析直接剪枝大设计空间）；(2) UP 上扩——遍历 bottleneck 循环，延迟仍 ≥ 最低延迟 n 倍者并行度提到 max(⌈n⌉×初始度, 最大度) 并迭代（n=2.0 经验值，匹配 unroll 最小粒度 2）；(3) DP 下缩——比最长循环快 n 倍的过度优化循环并行度除以 n 回收资源（可配置开关）。之后 inter-task optimization 把 bottleneck 循环策略传播给 FIFO 对端循环并重跑正确性 pass，冲突不可解处把该缓冲降级 ping-pong。效果：DSE 时间 0.1s–0.5s（StreamHLS MINLP 35s–20min 指数爆炸），仅 PA 阶段即 ZFNet 173.8×/YOLO 246.6×。与 AutoFHE（硬件参数×GA 调度）和 FlexQ-NDP（NDP 编译策略×DRAM 行切换代价模型）相比，CODO 的 DSE 空间是循环并行策略、代价模型为 profiling 参数化解析延迟估计、剪枝靠 FIFO 访问合法性分析。

PLENA 的 MOBO 视角（ISCA'26）：以多目标贝叶斯优化替代枚举/双层循环——搜索空间为硬件参数×量化精度的联合空间（BLEN/MLEN/VLEN/M_LOAD/V_LOAD/V_WRITE 为 2 的幂、ACT_WIDTH/KV_WIDTH∈{MXINT,MXFP}、FP_SETTING），目标三元组 f=(f_accuracy, f_latency, f_area)，用 BoTorch 多目标 BO 主动探索 Pareto 前沿，拒绝采样剔除非法候选（约束：MLEN·KV_WIDTH≤MemBandwidth、MLEN mod BLEN=0、MLEN≥HLEN≥BLEN）；结果用 EAS（Empirical Attainment Surfaces，多 seed 的 25%/75% attainment band）可视化。样本评估由多保真工具链完成：accuracy=H100 上跑 PTQ 的 perplexity、latency=transaction-level emulator、area=DC 综合模型——MOBO 的可负担性正来自这套多保真评估。实验：LLAMA3.2-1B 上 9 seeds×50 trials 对比 BoTorch vs TPE vs Random（BoTorch 显著更优），LLaMA-3-8B 上 5 seeds×50 trials（BoTorch vs Random）；最终选出 BLEN=32/MLEN=2048/VLEN=2048/W/A/KV=4/4/4 用于系统级评测。与 AutoFHE 的差异：AutoFHE 外层枚举硬件点 + 内层 GA 调度；PLENA 空间连续且有约束、三目标用 BoTorch 超体积类采集（qNEHVI/qEHVI 族，官方教程 20 批后 HV 显著优于随机），且目标含 accuracy 而非纯性能/面积。

Graph.hls 的分层 DSE 视角（ISCA'26）：DSE 空间 = 图加速器 L1/L2/L3 三层参数的组合，且按修改成本分层以支持跨层组合（与 AutoFHE 硬件点×GA、FlexQ-NDP 编译策略、CODO 循环并行策略均不同）：L3 数据流策略空间大且图拓扑相关（pipeline 分组需依据度分布/平均度），采用启发式选择（预测各候选配置的 pipeline 利用率，选最高）；L1/L2 空间在 L3 固定后由确定性双向依赖传播解析（Algorithm 1，Analyzer 过滤+选择，无启发式、无搜索）。设计意图 = 让原本不可组合的优化（位宽/分区比/pipeline 分组来自不同框架）经配置即可联合 DSE。效果（SSSP vs ReGraph 消融）：Naive 0.71×、L1-only 1.99×、L1+L2 2.95×（位宽双倍带宽+片上缓存）、L1+L3 2.52×（拓扑相关，偏爱倾斜图）、L1+L2+L3 4.48×——三层互补且 super-additive，验证"每层去除不同瓶颈"（L1 负载均衡、L2 带宽、L3 pipeline 形态）。GH-Scope 的 IR 级模拟/资源指标对比则作为 DSE 的快速评估通道（301.6× 于 Vitis C-Sim），区别于 AutoFHE 的解析模型。

NeRArch-Sim 的模拟退火 DSE 视角（ISCA'26，硬件参数空间 × SA 搜索 + 秒级 PPA 评估）：与上述硬件点×GA（AutoFHE）、编译策略（FlexQ-NDP）、循环并行策略（CODO）、三层图参数（Graph.hls）均不同，NeRArch-Sim 的 DSE 空间是神经渲染加速器的硬件模块配置——GSCore 案例中五个参数：(Culling Conversion Units, Quick Sorting Units, Bitonic Sorting Units, Volume Rendering Cores/VRCs, Buffer sizes)，目标为同时最小化 energy-delay product 与面积；用 Simulated Annealing 只评估设计点的一小部分（相对穷举 2.8× 加速），最优设计点 (16,8,4,32,4) 相对 GSCore 原配置 (4,8,4,64,8) 取得 1.3× energy-delay product 下降与 1.6× 面积下降（结论：VRC 过度配置应削减）。关键使能：端到端 PPA 评估秒级完成（表 XI：单设计点 47.7~79.2s，且硬件-only DSE 时算子图静态、instrumentation 复用，摊薄后每点仅调度器约 1 分钟），硬件特性离线预计算成查找表免去访问设计工具；DSE 还可缩放已有设计（表 X 的 CICERO+/GSCore-：把 CICERO 放大到 3DGS 级、GSCore 缩到相近 FPS，比较资源效率）。

  - SHyLA 补充（两阶段 DSE + 进化算法）：混合内存与部署设计空间过大，穷举代价高，故用"系统吞吐导向启发式 + 每用户吞吐约束进化探索"两阶段：(1) Stage 1 固定启发式部署（PD aggregation、(pt,pp)=(8,2)、MoE 加 pe=8），用解析模型在混合内存空间（Table III：7 种 NVM:DRAM 面积比 × 3 种 NVM 带宽-容量点 × 3 种 DRAM 点）穷举选 geomean 系统吞吐最优（结果 4:1 NVM:DRAM 面积比、NVM 平衡带宽-容量、DRAM 峰值带宽）；(2) Stage 2 用进化算法（population N=15、generation G=10、Top-K K=5）在部署空间（PD、各并行度、微批 b）中满足每用户吞吐 SLO（25 tok/s/user）下最大化系统吞吐，自适应进化方向（Algorithm 1）。DSE 输入 LLM workload 规格（如 GPT3-175B、(1024,1024)），运行于 96 核 Intel Xeon Gold 5418Y（Stage 1 约 3 分钟、Stage 2 约 8.5 分钟），选出 top 0.3% 配置；对 MHA/GQA、Dense/MoE 76 个 workload 验证泛化（MHA 优化点在 GQA 上仅 ~5% 退化）。
涉及论文标题：
- AutoFHE: An Automatic Hardware Generation Framework for Domain-Specific FHE Accelerators
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity
