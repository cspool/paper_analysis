## PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch

- baseline方法是什么？
  - Baseline 是标准 Expert Parallelism (EP) 下的 MoE 训练系统，包括 DeepSpeed、FasterMoE、Megablocks、Tutel、Janus。其核心执行模式为：每个 MoE layer 中 token 通过 gate network 路由后，执行两次 All-to-All 通信（dispatch token 到 remote expert 所在 worker，combine 计算结果回原 worker），通信占单层总时间的 50%-60%。FasterMoE 通过 shadowing/replicating 热门 expert 缓解负载不均，但 expert 参数的 periodic broadcasting 可能抵消 token 传输减少的收益。Janus 尝试在训练前优化 fetch expert vs send token 的决策，但当 expert 参数或 token 数据一方持续占优时失效。这些方法的共同缺陷是：(a) coarse-grained expert scheduling 与 All-to-All 通信同阶段执行，无法消除 All-to-All 瓶颈；(b) 仅支持 push-only 或 pull-only 范式，无法根据 token 分布动态选择最优数据传输方式；(c) backward pass 中 All-Reduce 和 All-to-All 争抢网络带宽，导致 computation blockage。
  - 全栈执行例子（Baseline: FasterMoE + MoE-GPT, ep=8, 8×RTX 4090, OpenWebText, top-k gating）：
    - **算法层**：Gating network（GShard 或 naive top-k）对每个 token 执行 Softmax(LinearGate(X)) → TopK(k=1/2) 选择 expert → 第一次 All-to-All（所有 worker 间全交换 dispatch token 到目标 expert 所在 GPU） → per-expert FFN 计算（两个 linear layer，GeLU 激活，H×αH → αH×H） → 第二次 All-to-All（combine expert 输出回原 worker）。FasterMoE 额外在所有 worker 上 shadow/replicate 热门 expert，通过 periodic broadcast 同步 expert 参数。All-to-All 占总时间 56%（单层 16 expert, batch 16）。
    - **系统框架层**：DeepSpeed-MoE / Megatron-LM 管理 EP（expert parallelism）+ DP（data parallelism）。各 worker 持有部分 expert，非 MoE 层（Attention）在 DP 组内 replicated。Communication backend 为 NCCL。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：PyTorch standard communication primitives（torch.distributed.all_to_all for token dispatch/combine），cuBLAS for expert FFN GEMM。两次 All-to-All 同步阻塞：expert computation 必须等所有 token 到达后才开始，combine 后必须等所有结果收集完才继续。热门 expert 承载更多 token 导致 compute skew，同时其所在 worker 的 network ingress/egress 量最大形成 network skew。非 MoE 层的计算期间 network link 完全 idle。
    - **硬件架构层**：Cluster A——2 节点 × 4 RTX 4090 24GB，节点间 100Gbps InfiniBand，节点内 PCIe 互联。All-to-All 通信走 InfiniBand（远慢于节点内），expert FFN 计算走 GPU SM。baseline 中 idle 链路（非 MoE 计算期间 InfiniBand 空闲）未被利用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 PopFetcher，通过 popularity-based expert-wise prefetching + hybrid push-pull + backward stream scheduling 三个核心机制，将 MoE 训练从"reactive scheduling（等 token 到达再调度 expert）"转变为"proactive preparation（预测热门 expert 并提前预取）"。
  - 全栈执行例子（PopFetcher, MoE-GPT, ep=8, 8×RTX 4090, sliding window s=10, OpenWebText）：
    - **算法层（预测 + 预取 — 解决"无法提前获知 expert 热度"缺陷）**：
      - Sliding-window popularity prediction：在每次 forward pass 中，routing information collector 记录各 token 在各 MoE layer 的 expert 选择 → 滑动窗口 s=10 iterations 内统计各 expert 的 token 分配比例 p_seq → 利用 expert 层间相关性，计算条件概率 Pr(E^{h,j+1}|E^{i,j}) = (1/M) Σ Pr(E^{h,j+1}|E^{i,j}, T_m) → 预测下一层 expert 流行度 p(E^{h,j+1}) = Σ Pr(E^{h,j+1}|E^{i,j}) p_seq^{i,j}（Eq. 3）。该计算在 CPU 异步执行，不影响 GPU training。对比 baseline 的 gate network 只能"事后知道" expert 分布——等当前层 router 执行完才知道哪些 expert 被选中。
      - Hybrid push-pull paradigm：不固定使用 push token 或 pull expert，而是根据公式对比——当 token 传输量 > 2048 tokens 时 pull expert（约 16MB for H=1024），否则 push token。对比 baseline FasterMoE 的 expert-only shadowing 和 Janus 的 pull-only 范式。
      - Expert prefetching decision formulation：建立 end-to-end training latency 模型 Lat_w^{prefetch}（Eq. 6），包含 forward pass computation time（local + prefetched expert）、backward pass computation time（×2）、token transfer time（仅未预取 expert 的 token 需 All-to-All）、gradient reduction time（prefetched expert 需额外 All-Reduce）。目标为 min max_w Lat_w^{prefetch}（Eq. 7）。
    - **系统框架层（预取决策 — 解决"coarse-grained scheduling 无法最优选择预取 experts"缺陷）**：
      - Expert prefetch pruning：两重约束剪枝搜索空间——(a) GPU memory limitation: 2αH² Σδ_{n,w}^i ≤ Mem_w^{free}（Eq. 8）；(b) Transfer time constraint: 2αH² Σδ_{n,w}^i/W_{n,w} ≤ Time^{non-MoE}（Eq. 9）。预取仅在计算-带宽比 ε = P_w/W_{n,w} > 3αH 时有效（Eq. 12-13），如 B200 + NVLink 400Gb/s 场景。被预取 expert 需满足 B_{n,w}^i > εαH / 2(ε-3αH)。按 popularity 排序预取 expert 直到 GPU memory 满。
      - Internal expert sharing via CPU memory：节点内 server-level cache manager 用 CPU memory 缓存已预取的 remote expert 参数 → 同节点其他 GPU 可直接从 CPU memory 读取，避免重复从 remote 拉取。优先通过 NVLink（1800GB/s）节点内检索，再由 GDR NIC（400Gb/s）跨节点拉取。
      对比 baseline 的 expert shadowing（无 memory-aware 决策）和 Janus（OOM on limited GPU memory）。
    - **编译框架层**：论文未明确说明。PopFetcher 实现为 PyTorch plugin（torch.autograd.Function 自定义 MoE operator），可集成到 Megatron-LM。
    - **kernel调度层（异步预取 + 流调度 — 解决"All-to-All 占关键路径"和"backward stream 争抢"缺陷）**：
      - Asynchronous prefetch execution：asynchronous scheduling executor 在 Attention 层（非 MoE 计算）期间，通过独立 CUDA stream 从 remote GPU 拉取已决策的 expert 参数 → 预取与当前层计算完全重叠，zero additional overhead on critical path。已预取到本地的 expert 的 token 直接本地计算——消除这部分 token 的 All-to-All dispatch/combine。
      - Stream pipelining in backward pass：将 All-to-All（token 回传）和 All-Reduce（prefetched expert gradient 聚合）分解为 micro-operations 交错流水线执行 → All-to-All 优先级高于 All-Reduce → 避免 All-Reduce 阻塞 All-to-All 导致 backward computation 等待。对比 baseline 三种通信（EP All-to-All + non-MoE All-Reduce + prefetched expert All-Reduce）启动三个独立 CUDA stream 并发时无优先级控制，network contention 导致 All-to-All 被延迟。
    - **硬件架构层**：同一 Cluster A/B 硬件。核心变化：baseline 中非 MoE 计算期间 InfiniBand/NIC 完全 idle → PopFetcher 利用 idle link 预取 expert 参数；baseline 中 All-to-All 占总时间 56% → PopFetcher 通过 token transfer 减少 14.85%（MoE-GPT）和 13.46%（MoE-BERT），GPU workload balance 提升（轻/重 worker token 差异减少 43.1% MoE-GPT, 57.1% MoE-BERT）；baseline 中 Janus 因 pull all experts 导致 OOM → PopFetcher 通过 pruning 约束内存，可训练模型尺寸比 FasterMoE 大 12.3%-20.1%，比 Janus 大 49.0%-58.2%。
  - 解决 Baseline 缺陷的方式总结：
    1. **All-to-All 通信瓶颈（占单层 50-60% 时间）**：通过 popularity prediction + asynchronous prefetching，在非 MoE 计算期间提前将热门 expert 参数拉到本地，使原本需 All-to-All dispatch 的 token 变为本地计算，减少 token 传输量 13-15%。
    2. **Coarse-grained expert scheduling（push-only 或 pull-only)**: Hybrid push-pull 根据 token 体积 vs expert 参数体积动态选择最优传输方式，当 token > 2048（H=1024）时 pull expert 否则 push token。
    3. **Backward pass network contention**: Stream pipelining 将 All-to-All 优先级置于 All-Reduce 之上，交错执行 micro-operations，减少 backward computation blockage 10-11%。
    4. **GPU memory 不足以 pull all experts**: Pruning strategy 基于 GPU memory capacity + transfer time budget 约束，优先预取 popularity 最高的 expert，middle-to-late training 阶段可固定预取方案或降低 replanning 频率。
