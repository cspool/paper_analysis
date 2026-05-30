## AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization

- baseline方法是什么？
  **人工/专家驱动的 AI 加速器 kernel 优化**：对于新兴 AI 加速器（如 Amazon Trainium），kernel 开发者依赖有限的性能直觉和优化经验手动调优 NKI kernel。初始 kernel 由 Neuron Compiler 自动生成，但其性能远低于硬件理论峰值（Trainium 1 上平均仅 49% peak throughput）。开发者需通过反复试验探索 loop ordering、tiling、memory layout、recomputation trade-off 等优化空间，过程耗时且依赖专家知识。

  全栈执行例子（以 NKIBench BatchMatmul+Softmax kernel 为例，Trainium 1 单核）：
  - **模型推理算法层**：Transformer 中 BatchMatmul 后接 Softmax 的标准算子链。Falcon-40B config: K=64, M=4096, N=4096。
  - **系统框架层**：Neuron Compiler 将高层 ML operator 图编译为 NKI kernel 调用序列。没有运行时调度框架的优化——kernel 优化完全依赖编译器自动生成的初始版本或人工手写的 NKI 代码。
  - **编译框架层**：Neuron Compiler 自动生成的 baseline kernel 分配 tiles 时，tile v 和 p 跨越两个嵌套循环存活，导致 SBUF 容量不足，触发 memory spilling（spill 到 HBM）。此外编译器不做跨循环的全局分析（如循环不变量外提），性能受限。
  - **kernel调度层**：Baseline kernel 使用固定 tile size（256 elements）和朴素 loop nest ordering（如 LHS transpose 在 i1 循环内重复执行 16 次）。HFU（Hardware FLOPs Utilization）仅 7.78%，memory write 达 1.07 GB。无自动的代数简化（如 θ-γλθ 仍为两次乘法和一次减法）或 intrinsic fusion。
  - **硬件架构层**：Trainium 1 单核 PeakMM 23.75 TFLOPS, PeakBW 440.2 GB/s, PeakVec 286.8 GFLOPS。Tensor/Vector/Scalar engine 并发运行，SBUF 每 partition 限 192KB，PSUM free dim 限 512。Baseline kernel 未充分利用 tensor engine（HFU 低），且大量 SBUF spilling 导致 HBM bandwidth 成为瓶颈。

  Baseline 核心缺陷：
  - (a) **优化空间巨大但探索效率低**：NKI kernel 需探索 memory layout、parallelization scheme、tiling、loop ordering 等多维度空间，人类需逐一尝试，时间成本高。
  - (b) **缺乏跨 iteration 的经验积累**：每次人工优化 kernel 从零开始，之前的优化经验无法系统性地迁移到新 kernel。
  - (c) **性能反馈周期长**：编译-运行-分析 loop 依赖人工介入，无法自动化规模化。
  - (d) **缺乏绝对性能标准**：传统 benchmark 只衡量相对 speedup，无法判断 kernel 是否已接近硬件理论峰值，优化方向不明确。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **AccelOpt：自改进 LLM agentic 系统**，通过 Planner-Executor-Summarizer 三代理工作流 + Beam Search + Optimization Memory，在无人工专家知识输入的情况下自主探索 NKI kernel 优化空间，迭代提升 kernel 性能。

  全栈执行例子（同样 BatchMatmul+Softmax kernel，Trainium 1 单核，AccelOpt B=6, N=12, K=2, T=16）：
  - **模型推理算法层**：同一 BatchMatmul+Softmax 算子链不变。AccelOpt 通过 Agent 理解算子语义（如识别 LHS transpose 在 i1 循环中不变），无需修改算法本身。
  - **系统框架层**：AccelOpt 在 Neuron Compiler 之上增加一层 meta-optimization 框架。Agentic workflow 不修改编译器，而是生成更优的 NKI kernel 源码作为编译器输入。分布式 profiling service 利用 Trainium core-level 和 machine-level 并行度批量测评 kernel。Roofline 模型计算每个 kernel 的 peak throughput percentage（T = max(Traffic_Min/BW, FLOPs_MM/Peak_MM, FLOPs_Vec/Peak_Vec)），提供绝对性能坐标系。
  - **编译框架层**：Agent 取代了人类专家的优化决策。Planner 分析 Neuron Profile 数据（HFU=7.78%, HBM write=1.07GB）识别 memory-bound 瓶颈 → 提出 "Hoist LHS Transpose Out of Reduction Loop" 优化计划 → Executor 实现 loop reordering（将 v7/v8/v9 transpose 外提到 i1 循环外，存入 global buffer v9_global，消除 16 次冗余计算）→ 进一步发现 recomputation 可消除 spilling（图 8: v 和 p tile 跨越两个循环导致 spill，通过 recompute v' 消除 spill，再通过移除额外 m loop 消除 recomputation 引入的额外 matmul）→ Summarizer 将 slow-fast kernel pair 提炼为通用经验 "Loop Invariant Code Motion for LHS Matrix Transposition"。
  - **kernel调度层**：AccelOpt 发现多种 kernel 级优化：(a) Peephole: θ-γλθ → (1-γλ)θ (代数简化), reciprocal(sqrt) → rsqrt (intrinsic fusion), x/(1+e^(-x)) → x·sigmoid(x) (利用 NKI 专用指令); (b) Loop 优化: tile size 256→512 (匹配 hardware optimal 128×512 moving 配置), loop fusion (将 i1 和 i5 循环融合为 16 次迭代的单个 i1 循环，减少嵌套开销)。优化后的 Mamba kernel 达到 54.6% peak throughput，超过人类专家最优版本 52.7%。
  - **硬件架构层**：同一 Trainium 硬件上，优化后的 kernel 通过降低 spilling (减少 HBM 访问)、提升 tensor engine utilization (HFU 从 7.78% 上升)、减少冗余计算，使 peak throughput 从 49% 均值提升至 61% (Trainium 1)。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: 优化空间巨大但探索效率低** → 方案: Beam Search 每轮生成 B×N×K 个 kernel，从 B+N×K 个候选中选择 Top-B 进入下一轮。B=6, N=12, K=2 → 每轮 144 个 kernel，T=16 轮共约 2304 个 kernel 被探索。比人类手动逐一尝试效率高数个量级。
  - **defect: 缺乏跨 iteration 经验积累** → 方案: Optimization Memory（容量 ExpN=16，每轮追加 TopK=8 items）存储 slow-fast kernel pairs + LLM-summarized 通用优化策略。正负样本（positive/negative rewrites with tpos=1.04, tneg=1.15 阈值）均收录。Memory 在候选组内做 diversity 过滤（group by candidate and plan，仅取异常值）。实验证明 memory 使达到相同 speedup 的迭代数减少 16-17%。
  - **defect: 性能反馈周期长** → 方案: 分布式 profiling service 利用 Trainium core/machine 级并行，B×N×K 个 kernel 可同时 profiling。Neuron Profile 提供详细硬件级指标（HBM read/write bytes, spill, engine utilization 等），直接输入 Planner 作优化依据。自动 correctness check（||output - cpuref|| < tol × ||cpuref||）过滤错误 kernel。
  - **defect: 缺乏绝对性能标准** → 方案: NKIBench 基于 Roofline 模型计算每个 kernel 的 hardware peak throughput percentage。Traffic_Min 为所有 I/O tensor 的 byte 总量，FLOPs_MM 和 FLOPs_Vec 分别计算 matmul 和非 matmul 操作的理论算力上限。绝对坐标系使系统能判断 kernel 是否已接近硬件极限（如 82% peak 后速度 plateau 是因为已接近理论峰值而非探索停止）。
  - **额外设计: cost efficiency** → Open-source 模型（gpt-oss-120b + Qwen3-Coder-480B）实现与 Claude Sonnet 4 相当的性能提升（61% vs 61% peak throughput on Trainium 1），但成本仅为其 1/26。Beam search 比重复采样更有效（图 13），optimization memory 提升 cost efficiency 但不过度影响最终最优 kernel 性能。
