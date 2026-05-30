## HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - HeterMoE 在 kernel 调度/运行时计算层面的核心实现：
    1. **Zebra Parallelism (ZP) 调度**：替代传统 Expert Parallelism (EP)。在一个 ZP group 内，expert 模块分布在 N 个 expert GPU（older generation），attention 等其余模块复制在 M 个 attention GPU（newer generation）。ZP 将每个 input batch 分为 R 个 microbatch，attention GPU 和 expert GPU 同时处理不同 microbatch，实现跨 GPU 的 compute-compute 重叠。同时，每 GPU 内维护 2 个通信 stream（dispatch + combine all-to-all）和 1 个计算 stream，通过 CUDA event 同步，实现 compute-communication 重叠。关键洞察：计算和通信重叠是利用了 dispatch 和 combine all-to-all 方向相反、不发生带宽竞争的特性。
    2. **Asymmetric Expert Assignment (Asym-EA)**：当 expert GPU 计算慢于 attention GPU 时（常见于短序列），将部分 expert 计算迁回 attention GPU。基于 "gather and squeeze" 算法（Algorithm 1）：accumulate 跨多层的气泡（bubble = T_E^Exp - T_A^Attn）直到足够 offload 至少一个 chunk 的 experts，然后 squeeze 气泡。最小 offload chunk 由 n_1 = max(1, N/M) 和 n_2 = n_1 · M/N 定义。考虑 memory 约束：通过 α 和 β 系数 enforce attention GPU 内存上限 n_max 和 expert GPU 内存下限 n_min。
    3. **Profiler**：测量 T_A^Attn（attention + gate 在 attention GPU 上的时间）、T_E^Exp（expert 在 expert GPU 上的时间）、T_E^Attn（expert 在 attention GPU 上的时间）。同时测量内存使用以估计 n_min 和 n_max。只需在每个 setup 上运行一次。
  - 实验比较：(1) HeterMoE vs EP (DeepSpeed MoE with Tutel/Lina optimizations) vs DistEP (naïve attention-expert disaggregation without overlapping) vs EP (Ideal, 各 GPU 型号独立运行后求和)；(2) HeterMoE vs heterogeneity-aware Pipeline Parallelism；(3) Ablation: GPU ratio in ZP group, fully homogeneous comparison, Asym-EA effects。

- 后端平台是什么，配置是什么。
  - **On-premise (O1/O2/O3)**：
    - O1: 6× A40 (48GB) + 6× V100 (16GB)
    - O2: 4× A40 (48GB) + 8× V100 (16GB)
    - O3: 6× A40 (48GB) + 3× V100 (16GB)
    - Network: 100 Gbps Mellanox ConnectX-6 RoCE NICs
  - **AWS (C1/C2)**：
    - C1: 2× L40S (48GB, g6e.4xlarge) + 6× T4 (16GB, g4dn.4xlarge)
    - C2: 2× L40S (48GB) + 8× T4 (16GB)
    - Network: 20 Gbps TCP (实际通信占 70% 训练时间，因此模拟 200 Gbps 通过减少 all-to-all 数据量实现)
  - 对比 homogeneous: 2× A100 (80GB, PCIe Gen4)

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch v2.2** (3K 行 Python) + **DeepSpeed v0.14**。
  - 修改内容：
    - **Zebra Parallelism Engine**：在 ZP group 内 split attention 和 expert 模块。初始化时创建 3 个 CUDA stream（2 个通信 + 1 个计算），为每个 microbatch 分配 receive buffer。创建独立的 NCCL dispatch 和 combine all-to-all group。通过 PyTorch NCCL all-to-all wrapper 传入不等 split size 实现不同 GPU 处理不同数量 tokens。
    - **Gate backward 修复**：gate network 的 top-k confidence scores 形成 "residual" 连接，backward 从 MoE block outputs 分两路传播（一路经 confidence scores 到 gate weights，另一路经 expert outputs 到 attention outputs）。HeterMoE 在每层 attention outputs 处停止第二分支的 backward，等待 expert GPU 梯度后再 accumulated 传播到前一层。
    - **Profiler**：从 transformer layer 提取单个 expert FFN，以实际 microbatch 对应的 token 数 B 生成 random tensor，分别在 attention GPU 和 expert GPU 上 profile forward+backward 时间。同时 profile memory usage 以估算 n_min 和 n_max。
  - 评估指标：training throughput (tokens/s)，GPU utilization（有效计算时间百分比），95% confidence intervals。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文声明 "We will open source HeterMoE"，截至分析时未找到公开代码仓库。
  - **Zebra Parallelism 执行原理全过程**：

```
┌── ZP Group Setup ──────────────────────────────────────────┐
│ M 个 attention GPU (newer, e.g. A40):                      │
│   - 持有复制: attention blocks + MoE gate + embedding      │
│   - 可选: 通过 Asym-EA 持有部分 experts                    │
│ N 个 expert GPU (older, e.g. V100):                        │
│   - 持有: expert FFN 模块（按 expert parallelism 分布）    │
│ R 个 microbatch: input batch 等分                           │
└────────────────────────────────────────────────────────────┘

┌── Stream Architecture (per GPU) ─────────────────────────┐
│ Stream 0 (compute):  attention/expert 计算                 │
│ Stream 1 (comm D):  dispatch all-to-all                    │
│ Stream 2 (comm C):  combine all-to-all                     │
│ Sync: CUDA events between streams                          │
│                                                             │
│ 例: attention GPU 上                                        │
│   Stream 1: enqueue dispatch A2A kernel                     │
│   event_dispatched = record()                               │
│   Stream 0: cudaStreamWaitEvent(event_dispatched)           │
│             → enqueue attention computation                 │
│             → ...                                           │
└────────────────────────────────────────────────────────────┘

┌── Forward Schedule (Theorem 1) ───────────────────────────┐
│ Attention GPU compute stream:                               │
│   (A_{1,1}^F ... A_{1,R}^F) ...                            │
│   (A_{L,1}^F A_{L,1}^B ... A_{L,R}^F A_{L,R}^B) ...       │
│   (A_{1,1}^B ... A_{1,R}^B)                                │
│                                                             │
│ Expert GPU compute stream:                                  │
│   (E_{1,1}^F ... E_{1,R}^F) ...                            │
│   (E_{L-1,1}^F ... E_{L-1,R}^F)                            │
│   (E_{L-1,1}^B ... E_{L-1,R}^B) ...                        │
│   (E_{1,1}^B ... E_{1,R}^B)                                │
│                                                             │
│ 其中 A_{i,j}^F/B: layer i microbatch j 的 attn fwd/bwd     │
│      E_{i,j}^F/B: layer i microbatch j 的 expert fwd/bwd   │
│                                                             │
│ 依赖约束（以 A_{i,j}^F 为例）:                              │
│   t(A_{i,j}^F) ≥ t(C_{i-1,j}^F) + T_C   (数据依赖)        │
│   |t(A_{i,j}^F) - t(A_{i',j'}^F)| ≥ T_A  (stream顺序)     │
└────────────────────────────────────────────────────────────┘

┌── Overlap Pattern (Zebra) ────────────────────────────────┐
│ Time →                                                      │
│ Attn GPU: [A_{1,1}^F][A_{1,2}^F][A_{1,3}^F][A_{2,1}^F]... │
│                                               ↕ overlap    │
│ Exp GPU:  [  E_{1,1}^F  ][  E_{1,2}^F  ][  E_{1,3}^F  ]...│
│                                                             │
│ 每 GPU 内 compute-communication overlap:                     │
│ Attn GPU: [Dispatch A2A][==== A^F ====][Combine A2A][A^F]  │
│ Exp GPU:  [==== E^F ====][Dispatch A2A][==== E^F ====]... │
│                                                             │
│ 关键：Dispatch和Combine走相反方向，在独立stream上不冲突     │
└────────────────────────────────────────────────────────────┘

┌── Asym-EA "Gather and Squeeze" (Algorithm 1) ────────────┐
│ Input: n (experts/layer), L (layers), M, N (GPU counts)    │
│        T_A^Attn, T_E^Attn, T_E^Exp (profiled times)         │
│                                                             │
│ n_1 = max(1, N/M)    // 每个 Attn GPU 至少 acquire 的 experts│
│ n_2 = n_1 · M/N       // 每个 Exp GPU 至少 offload 的 experts│
│ T_gather = T_E^Exp - T_A^Attn   // 每层每 microbatch 的气泡  │
│ T_squeeze = T_E^Exp·N/n·n_1 + T_E^Attn·N/n·n_2            │
│            // offload 一个 chunk 可消除的气泡                │
│                                                             │
│ t_bubble = 0                                                │
│ for l = 1 to L:                                             │
│   t_bubble += α·β·T_gather    // gather 气泡（含memory约束）│
│   if t_bubble ≥ T_squeeze:                                  │
│     chunks = floor(t_bubble / T_squeeze)                    │
│     o_l = chunks · n_2        // 该层 offload 的 expert 数  │
│     t_bubble -= chunks · T_squeeze                          │
│   else:                                                     │
│     o_l = 0                                                 │
│                                                             │
│ Memory约束:                                                 │
│   α = min(floor(n_max/n_2)·T_squeeze / (L·T_gather), 1)    │
│   β = max(ceil(n_min/n_2)·T_squeeze / (L·T_gather), 1)     │
│   (α和β 至多一个被激活，取决于offload量在上下界之间)       │
└────────────────────────────────────────────────────────────┘
```

  - **Profiler 评估原理**：
    1. 从 transformer layer 提取单个 expert FFN
    2. 根据 global batch size、seqlen、microbatch 数、ZP group setup 计算每个 expert GPU 处理的 token 数 B
    3. 生成 batch=B 的 random tensor → profile forward+backward 时间
       - 在 attention GPU 上: 得到 T_E^Attn
       - 在 expert GPU 上: 得到 T_E^Exp
    4. 提取 attention blocks + MoE gate → profile on attention GPU → T_A^Attn
    5. Memory profiling: 在 expert GPU 上构造单 expert FFN + dummy input → 测量 forward+backward 的 activation/weight/gradient/optimizer state 内存 → 估算 n_min（必须 offload 的 expert 数）；在 attention GPU 上构造不含 expert 的模型 → 估算 n_max（最多可持有的 expert 数）
  - **关键性能数据**：
    | Setting | HeterMoE vs EP | vs DistEP | vs EP(Ideal) |
    |---------|---------------|-----------|-------------|
    | O1/O2/O3 avg (4K) | +22% | +79% | +18% |
    | O1/O2/O3 avg (16K) | +67% | +69% | — |
    | O1/O2/O3 avg (32K) | +89% | +69% | — |
    | AWS avg | +189% | +96% | +17% |
    | vs homogeneous: 2A40+2V100 = 95% of 4xA40 throughput on avg |
