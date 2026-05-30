## HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

- baseline方法是什么？
  - **Tensor Parallelism (TP)、Expert Parallelism (EP)、Hybrid TP-EP with Compute-Balanced**。这三种策略是现有 GPU 集群和分布式系统上部署 MoE 模型的标准并行策略，但在 3D NMP 架构上各有根本缺陷。
  - 全栈执行例子（以 Mixtral-8x7B-Instruct 在 3D NMP 4×4 mesh 上推理为例）：
    - **模型推理算法层**：Mixtral-8x7B, 32 MoE layers, 8 experts/layer, top-2 gating。每个 token 经 gate 网络选择 2 个 expert，执行 gate_proj + up_proj + SiLU + down_proj（FFN 计算量 = 2h * IS = 2*4096*14336 FLOPs/token/expert）。
    - **系统框架层**：
      - **TP baseline**：每个 expert 的权重沿 intermediate dimension 切分到所有 16 个节点。每层执行：各节点计算 expert FFN 的 1/16 输出 → ring all-reduce 聚合结果（通信量 = 4Bh/BW per layer，其中 B 是 batch size，h 是 hidden dim）。计算负载完全均衡，但 all-reduce 通信随 batch size 线性增长，在 3D NMP 的有限 NoC 带宽下成为瓶颈。
      - **EP baseline**：每个 expert 完整分配给一个节点（8 experts 分布在 16 个节点，部分节点空闲）。token dispatch 经 all-to-all 通信送到对应 expert 所在节点 → 节点本地计算完整 expert FFN → all-to-all combine 聚合结果。通信量小（仅 token hidden states 传输），但 expert 激活频率不均衡导致计算负载倾斜（hot expert 所在节点成为瓶颈）。
      - **Hybrid TP-EP with Compute-Balanced baseline**：2D mesh 划分为子区域（mixtral 2 子区），区域内 TP + 区域间 EP。根据 expert 平均激活频率静态分配 expert 到子区域以平衡计算量。但忽略物理拓扑对通信的影响，在带宽受限下性能退化。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：EP 的 all-to-all dispatch/combine（NCCL 等价原语）在 2D mesh 上产生不规则通信模式；TP 的 ring all-reduce 是结构化通信但数据量大。静态调度无运行时适应能力。
    - **硬件架构层**：3D NMP 加速器，每个节点有独立 local memory bank + compute die，通过 2D mesh NoC 互联（无 shared memory / shared L2 cache）。节点间通信通过 NoC 链路（configurable bandwidth 25-75 GB/s per link）。
  - Baseline 痛点：
    1. **TP 通信瓶颈（3D NMP 特有痛点 1）**：3D NMP 架构无共享内存，TP 的 all-reduce 通信必须经过 2D mesh NoC 的有限带宽链路。通信量 = 4Bh/BW 随 batch size 线性增长，在大 batch 或低带宽配置（10 TFLOPS, 25 GB/s）下成为严重瓶颈。
    2. **EP 负载不均衡（3D NMP 特有痛点 2）**：MoE 的 expert 激活天然不均衡（如 Qwen2 中近半 token 汇聚到单一 expert），EP 将完整 expert 分配给单一节点，导致 hot expert 所在节点成为计算瓶颈，其他节点空闲。3D NMP 的分布式内存无法像 GPU 集群那样通过复制 hot expert 来缓解（内存容量受限）。
    3. **Hybrid TP-EP 忽视物理拓扑（3D NMP 特有痛点 3）**：现有 hybrid 策略只考虑逻辑层面的 compute balance，忽略物理 2D mesh 拓扑中链路级拥塞。计算均衡的 placement 可能产生集中通信路径，导致链路热点和通信 tail latency。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HD-MoE 方法**：通过 Offline Automatic Hybrid Parallel Mapping + Online Dynamic Scheduling 两阶段设计，在 3D NMP 架构上协同优化 MoE 推理的计算均衡和通信效率。
    1. **Performance Analytical Model**（解决痛点 1、2、3 的基础）：构建统一性能分析框架，computation 模型基于 per-node token load 取 max，communication 模型通过 discrete-event simulation（XY routing + priority queue scheduling）精确建模 2D mesh 上不规则 all-to-all 延迟。推导线性近似 t̂_comm = (4/BW) * max_c{ Σ_g (Π_{i∈g} ⌈P_ic⌉) * f_g * B * h }，经验验证 R² > 0.9，为 LP 优化提供可微目标函数。
    2. **Node-Link Balance Co-optimization**（解决痛点 1、2、3）：
       - **Stage 1 - Node Balance (LP)**：将 expert placement 形式化为线性规划问题。连续变量 P_ic ∈ [0,1] 表示 expert i 分配到节点 c 的比例（允许部分分配，即 TP 模式）。目标 min(t_comp + 2γ*t̂_comm)，约束包括：(a) 每个 expert 分配完整（Σ_c P_ic = 1），(b) 计算负载有界（基于 compute/communication ratio R_CC），(c) 通信量均衡。**关键设计**：P_ic 是连续值而非 binary，允许 hot expert 部分切分（TP 模式）以平衡计算，cold expert 完整分配（EP 模式）以减少通信。这实现了 TP-EP 的自动混合。
       - **Stage 2 - Link Balance (Bayesian Optimization)**：将 LP 得到的逻辑集群映射到 2D mesh 物理节点。目标是最小化链路拥塞和通信 tail latency。Bayesian Optimization 适合此问题因为：(a) 每次评估需运行 discrete-event simulation（expensive），(b) 目标函数相对平滑（相邻节点交换只引起微小通信成本变化）。
    3. **Dynamic Placement Strategy**（解决 EP 的运行时负载不均）：
       - Priority Detection：利用相邻层 expert activation 的时间局部性预测下一层热点，计算 per-expert 优先级 = 2*P_ic*f̂_i*IS/comp。
       - Optimal Pre-broadcast：对预测热点 expert，在上层推理进行时利用空闲 NoC 带宽将其预广播到所有节点。α-β 模型推导最优 chunk size c = sqrt(α*h*IS/(2*β*k*sqrt(D)))。
       - Communication-Efficient Dispatch：预广播后 token 路由到持有其激活 expert 的节点中负载最低者，避免额外通信。
  - 全栈执行例子（HD-MoE, Mixtral-8x7B, 3D NMP 4×4 mesh，与 baseline 同配置对比）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B, top-2 routing），不修改模型结构或 gate 逻辑。
    - **系统框架层**：Offline 阶段运行 LP + BO 搜索最优 P_ic 和物理映射。对 Mixtral 的 8 experts：hot expert（如 expert 0, 3）部分切分到多个节点（TP mode, P_ic < 1），cold expert 完整分配到单一节点（EP mode, P_ic = 1）。Online 阶段：每层推理前运行 Priority Detection，预测下一层热点 → 在上一层计算进行时预广播热点 expert（利用空闲 NoC 带宽）→ 每个 token dispatch 到负载最低的候选节点。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：对比 baseline 的静态 all-to-all（EP）或 all-reduce（TP），HD-MoE 的通信模式是 hybrid——hot expert 的分片节点间进行 TP 式 all-reduce（小范围），cold expert 保持 EP 式 all-to-all（稀疏）。Dynamic 阶段在上层计算时异步预广播热点 expert 权重（利用 α-β 最优 chunk size），dispatch 时基于节点实时负载做贪心路由。
    - **硬件架构层**：与 baseline 相同的 3D NMP 4×4 mesh。结果：speedup 1.1-1.8× vs TP，1.1-1.5× vs EP，1.0-1.4× vs Hybrid TP-EP。关键硬件利用策略：(a) 计算受限（2.5 TFLOPS, 75 GB/s）时，HD-MoE 自动偏向 EP 为主的混合策略，减少 TP all-reduce 开销；(b) 通信受限（10 TFLOPS, 25 GB/s）时，自动偏向 TP 为主的混合策略，避免 EP 的 all-to-all 不规则通信造成链路拥塞。
