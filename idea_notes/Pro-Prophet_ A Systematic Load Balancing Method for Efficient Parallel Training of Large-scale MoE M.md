## Pro-Prophet: A Systematic Load Balancing Method for Efficient Parallel Training of Large-scale MoE Models

- baseline方法是什么？
  - Baseline 是两种代表性 MoE 训练框架：(1) DeepSpeed-MoE (Microsoft)——纯 Expert Parallelism (EP) 训练框架，无 load balancing 优化；(2) FasterMoE——系统性 load balancing 方法（dynamic shadowing），通过动态调整 expert placement 来平衡负载，但引入三个核心缺陷：Search（在 runtime 搜索 load balancing 策略，占训练时间 3-7%）、Place（传输 expert 的 parameters 和 gradients 到所有 devices，占 12-16%）、Reduce（将梯度汇聚回原 device，占 12-18%），总 load balancing 开销最高达 37.1%。
  - 全栈执行例子（Baseline: FasterMoE, MoE-GPT-M, 16 GPU, HPWNV 集群）：
    - **算法层**：Gating network 对每个 token 执行 Softmax(LinearGate(X)) → TopK(k=1/2) 路由到 expert。FasterMoE 的 dynamic shadowing 方法：在 runtime 检测各 device 的负载 → search 最优 expert placement → 将 heavy-load expert 的 parameters 传输到 light-load device → 完成后执行 expert computation → 反向传播后将 gradients 聚合回原 device。Baseline 的核心缺陷在算法层：heavy-load expert 的 parameters/gradients 需要在所有 devices 之间全局传输（而非仅必要 device 子集），通信量巨大；且 search 过程本身耗时。
    - **系统框架层**：Expert Parallelism 将 experts 均匀分配到各 device，非 MoE 层（Attention）复制到所有 device。每次 MoE layer 执行：gate → All-to-All dispatch（将 token 按 routing 发送到对应 expert 所在 device） → expert FFN → All-to-All combine（将输出返回原 device）。FasterMoE 额外插入 search→place→reduce 流程。PyTorch Distributed + NCCL backend。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：All-to-All 通信使用 Tutel 的高效 P2P 实现。Expert FFN 计算使用标准 cuBLAS GEMM。关键问题：search/place/reduce 与 expert computation 之间存在数据依赖——必须在 gate 输出后才能 search placement，必须在 placement 确定后才能 place（传输参数），必须在 backward 完成后才能 reduce（聚合梯度）——导致这些操作串行执行无法被隐藏，产生大量 communication idle 和 computation idle。
    - **硬件架构层**：NVIDIA 3090 GPU (24GB) × 16，PCIe 3.0 连接，节点间 100Gb/s Infiniband。关键缺陷：place 阶段将 heavy-load expert 参数传输到所有 device 导致不必要的跨节点 Infiniband 通信；reduce 阶段类似；且这些通信在时间线上与计算串行无法重叠。以 MoE-GPT-M 为例，load balancing 开销占总训练时间 29.2%（含 search 3.2% + place 12.5% + reduce 12.5%）。
  - Baseline 核心缺陷根因（两个）：
    1. **Heavy communication of model states**：FasterMoE 的 expert placement 采用全局传输——heavy-load expert 的参数/gradients 在所有 devices 之间传输（而非仅传输到该 expert 有 input 的 device 子集），导致大量不必要的跨节点通信。
    2. **Poor communication-computation overlapping**：由于数据依赖（必须先 search 才能 place，必须先 backward 才能 reduce），place/search/reduce 操作串行化在关键路径上，无法与 computation 重叠，大量通信和计算时间 idle。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 Pro-Prophet，包含 Planner + Scheduler 两个核心组件。核心 insight 是发现 MoE 训练中相邻迭代的 input distribution 存在 locality（高度相似），使得可以预测未来的分布并据此提前做 scheduling。Planner 解决 heavy communication 问题（通过 lightweight expert placement 仅传输到必要 device 子集 + greedy search 找 communication-efficient placement），Scheduler 解决 poor overlapping 问题（通过 block-wise scheduling 将 load balancing 操作与 computation 重叠）。
  - 全栈执行例子（Pro-Prophet, MoE-GPT-M, 16 GPU, HPWNV 集群）：
    - **算法层（Planner — 解决 heavy communication 缺陷）**：
      - **Lightweight Expert Placement**：每个 expert 独立映射到必要的 device 子集（而非全部 devices）。例如 expert E_0 的 input 分布在 device 0 和 device 1（device 2 没有 E_0 的 input），则 E_0 的 parameters 仅从 device 0 传输到 device 1（不传输到 device 2），gradients 也仅在 device 1 传回 device 0。对比 baseline FasterMoE 将 expert 参数传输到所有 devices。
      - **Performance Model**：抽象公式化 MoE 层执行时间——T'(R,H,s,n) = 4·T_A2A(R) + 3·T_FEC(H) + T_Trans(s,n) + T_Agg(s,n)。mean estimation error <5%。
      - **Locality-based Greedy Algorithm**：利用 locality 减少 search 频率（用户可调节）。算法在 runtime 贪心迭代——每次选择当前负载最重的 device 上的 expert，将其 parameters 传输到持有该 expert 最多 input 的 devices 子集——每次迭代用 performance model 评估 placement，直到负载满足 max(H) - min(H) < α·I/E 的平衡条件。因搜索空间为 2^(N·E) 的 brute-force 不可行，greedy 策略使 search 可行性成立。
      - 对比 baseline：FasterMoE 的 search 是全局传输策略（heavy-load expert → 所有 devices），通信量 O(D·size(expert))；Pro-Prophet planner 的 lightweight placement 仅传输到必要 devices 子集，通信量 O((D-n)·size(expert))，其中 n 是不必要的 device 数——当 n 大时通信量显著降低。
    - **系统框架层（Scheduler — 解决 poor overlapping 缺陷）**：
      - **Scheduling Space 建立**：定义每个 MoE block 内的操作类型（comm vs comp）和数据依赖。利用 locality 提前预测 iteration j+1 的 input distribution → Plan_{i}^{j+1}（决定 j+1 迭代的 placement）可在 iteration j 的 A2A 通信中执行。Trans 原语（传输 expert 参数）可在同一 iteration 内与 forward computation 重叠。Agg 原语（聚合 gradients）可在同一 iteration 内与 backward computation 重叠。Scheduling 约束：Plan 必须在上一迭代执行（需要上一迭代的 distribution 数据）；Trans 和 Agg 各自限制在单个 iteration 内（因 layer-by-layer 和 concentrated updating 两种参数更新方式兼容性）。
      - **Block-wise Scheduling Strategy**：以 MoE block 为单位进行 sub-operator 级调度。将 Trans 原语拆分为 2 个 sub-operators，分别与同一 block 的 FEC（Forward Expert Computation）和 FNEC（Forward Non-Expert Computation）并行。类似地，Agg 原语与 BEC 和 BNEC 重叠。FNEC 和 BNEC 的时间是静态的（可在训练前估计），用于精确规划 split。Plan 操作的 sub-operators 被调度到前一迭代的 A2A 通信中执行。
      - 对比 baseline：FasterMoE 的 search/place/reduce 串行暴露在关键路径上。Pro-Prophet scheduler 将 Plan（search）隐藏在前一迭代的 A2A 通信中，将 Trans（place）隐藏在 forward computation 中，将 Agg（reduce）隐藏在 backward computation 中——所有 load balancing 开销被计算时间覆盖。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel 不变。关键变化在数据流和执行时序：baseline 中 search→place→reduce 与 compute 串行——gate → search → place (transmit params) → A2A → compute → backward → reduce (aggregate grads)；Pro-Prophet 中 Plan 提前到前迭代执行（与 A2A 并行），Trans 与 FEC+FNEC 并行，Agg 与 BEC+BNEC 并行——gate → A2A (含 Plan_{next}) → [compute FEC || Trans sub-op1] → [compute FNEC || Trans sub-op2] → ... → backward [compute BEC || Agg sub-op1] → [compute BNEC || Agg sub-op2]。
    - **硬件架构层**：同一 NVIDIA 3090 × 16 + Infiniband 硬件。核心变化：baseline 中 Infiniband 链路在 compute 期间 idle（search/place/reduce 独占通信链路但串行执行），Pro-Prophet 下 Infiniband 持续在 compute 期间并行传输 expert 参数和 gradients（Trans/Agg 与 FEC/BEC 重叠），链路利用率提升。另外 locality 减少了 search 频率，进一步降低通信开销。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对 heavy communication of model states**：Lightweight expert placement 将 expert 仅映射到有其 input 的 device 子集（而非全部 devices），Trans 和 Agg 仅在子集内执行。Performance model + greedy algorithm 在 runtime 高效搜索 communication-efficient placement。对比 FasterMoE 全局传输策略，communication volume 因 device 子集缩小而显著降低。
    2. **针对 poor communication-computation overlapping**：Block-wise scheduling 利用 locality 预测未来分布，将 Plan 提前到前迭代的 A2A 中执行，将 Trans 与 forward computation 重叠，将 Agg 与 backward computation 重叠——所有 load balancing 开销被计算隐藏。对比 FasterMoE 串行执行导致 29-37% 的 load balancing 开销，Pro-Prophet 通过重叠将这些开销几乎消除。
    3. **关键数据支撑**：Planner 单独贡献 1.12-1.26x 加速，Scheduler 单独贡献 1.01-1.14x 加速，Full 协作额外贡献 1.02-1.03x 加速。vs FasterMoE 的 load balancing 提升（RB ratio）最高达 11.01x。Performance model estimation error <5%，验证了建模精度。
