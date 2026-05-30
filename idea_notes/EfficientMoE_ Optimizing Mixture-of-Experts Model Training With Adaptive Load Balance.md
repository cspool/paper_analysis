## EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance

- baseline方法是什么？
  - Baseline 是静态图模式下基于 expert parallelism 的标准 MoE 训练（如 Switch Transformers 的 MindSpore 静态图移植、Fastermoe 的 MindSpore 算子迁移）。静态图要求输入 shape 在编译前固定，所有 expert 共享同一 capacity，无法在运行时动态调整。
  - 全栈执行例子（Baseline: Switch Transformers 在 MindSpore 静态图, 32 Ascend 910, Expert Parallelism = 16, MoE-θ 21B）：
    - **算法层**：Gating network 对每个 token 执行 Softmax(W_gate · x) → TopK(k=1) 路由到最相关 expert。各 expert 的 FFN 独立处理分配给它的 token 子集，输出按 token 聚合。在大规模 MoE 中，token 分布极度不均匀——约 70% token 集中于 2 个 hot experts，其余冷 experts 仅处理少量 token。
    - **系统框架层**：MindSpore 2.0 静态图模式。使用 DP=16、MP=2、EP=16 的混合并行策略。32 个 expert 分布在 16 个 expert-parallel 组中的 accelerator 上。每个 iteration 中，All-to-All 通信在 dispatching 阶段将 token 从源 accelerator 发送到对应 expert 所在的 accelerator，在 combining 阶段将处理后的 token 回传。
    - **编译框架层**：MindSpore 静态图编译器在训练前编译整个计算图。所有 expert 使用相同的固定 capacity（由编译时预定义的最大 token 数决定），无法在运行中改变。hot expert 的输入超出 capacity 时剩余 token 被直接丢弃；cold expert 的输入远小于 capacity 时需填充 zero vectors 至固定 shape。
    - **kernel调度层**：Ascend 910 上执行 MindSpore 算子。All-to-All 通信占纯通信时间的 75%（论文 profiling 分析，32 加速器集群）。Expert FFN 计算时，cold expert 因大量 padding 浪费约 50% 的计算资源——zero vectors 参与的矩阵乘法结果被丢弃但计算已执行。
    - **硬件架构层**：4 节点 × 8 Ascend 910，节点间 100 GB/s RoCE。节点内 8 卡共享高速互联带宽。All-to-All 跨节点通信成为瓶颈——32 加速器集群中纯通信时间占 MoE 训练总时间的 75%，大量 AI accelerator 计算周期因等待通信而闲置。
  - Baseline 缺陷根因（两个核心问题）：(1) **负载不均 + All-to-All 通信开销**：token 分布的极端不均衡（70% token 集中在 2 个 expert）导致 hot expert 过载/cold expert 空转，同时 expert parallelism 要求将 token 跨节点发送到 expert 所在 accelerator 处理，All-to-All 通信占比高达 75%，训练效率受通信瓶颈限制；(2) **静态 capacity 下的 token 丢弃与 padding 浪费**：静态图要求编译前固定 expert capacity，hot expert capacity 不足导致 token 丢弃影响精度，cold expert 因 capacity 远大于实际负载而大量 zero-padding，浪费约 50% 计算资源。这两个问题在动态图框架（PyTorch）下有成熟的解决方案（如 Fastermoe 的动态 shadowing、Megablocks 的动态 capacity），但在高效的静态图框架（MindSpore）下缺乏对应优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：EfficientMoE 在 MindSpore 静态图框架内，通过三阶段优化解决上述问题——(1) Sampler + Load Prediction Model 实时监控并预测 expert 负载；(2) Dynamic Expert Scheduling 将 hot expert replica 部署到冷 expert 所在 accelerator，本地化 token 处理；(3) Expert Capacity Model 在编译前为 hot/cold expert 设置差异化 capacity。
  - 全栈执行例子（EfficientMoE, MoE-θ 21B, 32 Ascend 910）：
    - **算法层**：
      - Sampler 在每个 iteration 收集 token 分布、expert 参数、内存和通信四维信息（Table II）。
      - Load Prediction Model 在 m 次 iteration 的 cycle 中统计各 expert token 负载 L_i^k = mean(Top_p%(sorted(token_counts)))，并计算 accelerator 负载 D_j = Σ(Compute(T_i) + Memory(T_i))（公式 1-2）。
      - Expert 按负载因子与阈值 q=60% 比较，分为 hot（负载 > q·max_load）和 cold（负载 ≤ q）。
      - Dynamic Schedule（Algorithm 1）：为 hot expert 生成 replica，调度到有空闲资源的 cold expert 所在 accelerator。带来的关键变化：原本需跨节点 All-to-All 发往 hot expert 的 token，现在被本地 replica 直接处理，仅需同步 replica 参数更新（体积远小于 token 传输量）。
      - Expert Capacity Model（公式 3-6，Algorithm 2）：C_j^i = (1-r)·B + r·(1/m)·ΣF_t，其中 B 为统计 baseline，r 由峰值负载决定（公式 4），F = γ·(T_i-B)/Total_tokens（公式 5）。hot expert C_j^i > B（增大 capacity 减少 token 丢弃），cold expert C_j^i < B（减小 capacity 减少 padding）。总内存需满足 M_total ≤ accelerator_memory（公式 6）。
    - **系统框架层**（MindSpore 2.0 + Mindformers 1.0 修改）：
      - 在 MindSpore 静态图内核中插入 Sampler 模块，在每个 training iteration 的 MoE 层前拦截并收集 token 路由信息。
      - 在静态图编译前，Load Prediction Model 读取 sampler 历史数据，输出 hot/cold 分类和 per-expert capacity 设置。
      - Dynamic Scheduling 模块在 load prediction cycle 边界修改 MindSpore 的 expert placement（将 hot expert replica 部署到目标 accelerator），并调整 token routing 逻辑——gate 后的 token routing 表被修改为：若有本地 replica 则路由到本地，否则保留 All-to-All 到原 expert。
      - 对比 Baseline 的"固定 expert placement + 全量 All-to-All"，EfficientMoE 变为"动态 replica placement + 部分本地处理 + 剩余 All-to-All"。
    - **编译框架层**：MindSpore 静态图编译器。EfficientMoE 不修改编译器内部，而是在编译前注入差异化的 expert capacity（C_j^i）以替代原本统一的固定 capacity。编译后每个 expert 的输入 buffer shape 已按各自 capacity 分配，hot expert buffer 更大（接收更多 token），cold expert buffer 更小（减少 padding）。capacity 调整在 cycle 间触发 re-compilation（论文未详细说明 re-compilation 策略的具体开销）。
    - **kernel调度层**：Ascend 910 上算子执行逻辑不变。关键差异在数据流：(a) Baseline 中每个 iteration 有大量 token 通过 RoCE 跨节点 All-to-All 传输到 hot expert，通信占据了 75% 时间；(b) EfficientMoE 中，因 hot expert replica 本地化了大量 token——原本需跨节点发送的 token 数据流变为本地 NVLink/内存拷贝，仅剩少量 cold expert token 需 All-to-All。同时 cold expert 的 zero-padding 因 capacity 减小而大幅削减，减少了约 35% 的无效计算量。
    - **硬件架构层**：同一 4 节点 Ascend 910 集群。核心变化：All-to-All 通信从占迭代时间的 75% 显著降低（约 12% 通信时间减少），计算资源浪费从约 50% 降低（35% 计算资源节省），综合训练时间缩短 30%。RoCE (100 GB/s) 的带宽瓶颈因 All-to-All 流量减少而缓解，accelerator 计算利用率因减少 padding 而提升。
  - 解决 Baseline 缺陷的方式：
    1. **针对"负载不均 + All-to-All 通信瓶颈"**：EfficientMoE 用"expert replica 调度"替换"token 跨节点路由"。核心转换——将大体积高频的 token All-to-All 通信变为小体积低频的 expert 参数同步（replica 参数更新在 load prediction cycle 边界同步一次而非每 iteration）。这类似于计算与数据的 co-location 优化——不是"把数据送给计算"，而是"把计算搬到数据所在地"。在 4 节点 32 accelerator 集群上实现约 12% 通信时间降低、30% 端到端加速。
    2. **针对"静态 capacity 下的 token 丢弃与 padding 浪费"**：EfficientMoE 用 Expert Capacity Model 在静态图编译前的 cycle 边界为各 expert 设置差异化 capacity——hot expert replica 获得更大 capacity（减少 token 丢弃，保护精度），cold expert 获得更小 capacity（减少 zero-padding，节省 35% 计算资源）。capacity 的周期性重评估（每 m 次 iteration 基于 token 分布变化触发）解决了"静态 capacity 无法适应负载变化"的核心矛盾。
    3. **针对"动态图优化方法无法用于静态图"**：EfficientMoE 将所有优化保持在静态图范式内——load prediction cycle 的周期性评估 + 编译前 capacity 注入 + replica placement 修改——不依赖动态图运行时的 shape 变化能力。这使得 MindSpore/Ascend 生态的 MoE 训练能同时享受静态图的计算效率优势和 EfficientMoE 的负载均衡优化。
