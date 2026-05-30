## Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Lazarus 在 kernel 调度/运行时计算层面的核心实现：
    1. **Flexible Token Dispatcher CUDA Kernel（Algorithm 1）**：将整个 MoE 层的 token dispatch 实现为单个 CUDA kernel，对所有 experts 和 target ranks 并行处理。核心逻辑：(a) 根据 expert 的 replica 数 r_e 和每 replica 应处理的 token 数 p_e = t_e/r_e，计算每个 rank 对每个 expert 的处理容量 P_{e,j} = p_e × R_{e,j}；(b) 优先将 rank j 本地已有的 token 分配给它自身（min(P_{e,j}, T_{e,j})）；(c) 将超出本地容量的剩余 token 按各 rank 的剩余容量比例分配到其他 rank（(T_{e,i} - D_{e,i}) × P_{e,j} / Σ_k P_{e,k}）；(d) 根据 dispatch schedule 将输入 activation h 重排（reshuffle）为连续 buffer h'，使 routed to same expert + dispatched to same rank 的 token 连续排列，供后续 all-to-all collective 使用。
    2. **Adaptive Expert Replica Allocation（Eq. 1）**：运行时根据 expert load 分布动态计算每个 expert 应分配的 replica 数 r_e = max{⌊t_e / Σ_{e'=e}^{E} t_{e'} × (N·c - Σ_{e'=1}^{e-1} r_{e'})⌋, f}，其中 t_e 为 routed token 数，N 为节点数，c 为每节点 replica 槽位数，f 为容错阈值。同时保证 Σ_e r_e = N·c, r_e ≥ f（支持 <f 个节点故障时 100% 恢复）。
    3. **Maximum Rank Overlap (MRO) 专家放置算法**：将 experts 按流行度分为 ⌈E/c⌉ 组，每组内最大化 experts 跨节点的重叠度（S_{c*(i-1)+1} ⊂ S_{c*(i-1)+2} ⊂ ... ⊂ S_{c*i}），使某组 representative expert 的 replica 存在即可恢复全组。定理证明 MRO 在均匀随机节点故障下最大化恢复概率。
    4. **Efficient Reconfiguration Runtime**：故障后利用贪心算法最小化迁移的 replica 数，将物理节点映射到新 placement plan 中重叠度最大的节点，跨节点并行获取缺失 expert states。
  - 实验比较：Lazarus vs DeepSpeed MoE (DS)、DS(FT)（使用 Lazarus runtime 的容错版本），以及 Tutel、Tutel(FT) 在 controlled failures（单节点/多节点）、spot instance traces 下的吞吐量和总训练样本数。消融实验：单 MoE layer 在不同 expert load ratio 下的吞吐量对比 + 恢复概率对比。

- 后端平台是什么，配置是什么。
  - **本地集群**：5 台服务器，每台 2× NVIDIA RTX 3090 GPU + 100 Gbps Mellanox ConnectX-5 NIC，100 Gbps Mellanox SN2100 switch。每 GPU 视作独立节点模拟 10 GPU 集群。NFS server 通过 10 Gbps NIC 连接用于 checkpoint 存储。
  - **AWS 集群**：16× g5.2xlarge instances，每实例 1× NVIDIA A10G GPU，10 Gbps TCP 网络。AWS EFS 共享文件系统存储 checkpoint。使用 gradient accumulation (step=20) 减少频繁梯度同步。
  - **模拟环境**：模拟 DeepSeek V3 模型训练，每节点 8× H200 GPU + 8× 400 Gbps NIC。使用 DeepSeek-V3/R1 Performance Simulator (github.com/zartbot/shallowsim) 的性能模型。
  - 软件：PyTorch v2.3，DeepSpeed v0.13（组件复用），CUDA，NCCL v2.12.12。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **PyTorch** (v2.3) 实现，使用 **DeepSpeed** (v0.13) 组件。总计 4K LoC Python + 500 LoC CUDA。
  - **修改/新增内容**：
    1. **Flexible Token Dispatcher CUDA Kernel (~500 LoC CUDA)**：实现 Algorithm 1 的 CUDA kernel，在 MoE block forward path 中替代原有 DeepSpeed MoE 的 dispatch 逻辑。计算 dispatch schedule → reshuffle input activations → 执行 flexible all-to-all（无 padding）。
    2. **Controller + Agent 架构（Python async）**：Controller（CPU-only 节点）管理集群全局状态，通过 TCP socket 与各 GPU 节点的 Agent 通信。Controller 执行 MRO placement 算法（<100ms 计算时间），Agent 周期性收集 expert routing history 并 relay placement plan 给 worker。
    3. **Lazarus Runtime**：基于 placement plan 配置 NCCL communication groups（expert/non-expert gradients all-reduce + all-to-all）。Data Parallelism + Expert Parallelism with adaptive placement。Batched NCCL send/recv 用于 reconfiguration 时的 state transfer。
    4. **Routing History Trace Replay**：使用 SmartMoE artifact 的 routing history trace 模拟 gate network routing decision，保证可复现性。
  - 评估指标：training throughput (samples/sec)，total trained steps/samples，reconfiguration time，state transfer size/time，recovery probability（理论枚举所有故障组合）。
  - 模拟评估：自建 simulator，使用 constant node preemption probability + variable new node allocation probability per simulation hour。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文声明 "We will open source Lazarus"，截至分析时（2026/05）未找到公开代码仓库。
  - **Flexible Token Dispatcher CUDA Kernel 执行原理全过程**：
```
┌── Kernel Input ──────────────────────────────────────────┐
│ N: number of GPUs; i: current GPU rank                    │
│ R_{e,j}: number of replicas for expert e assigned to     │
│         rank j (from placement plan)                      │
│ T_{e,j}: number of tokens routed to expert e at rank j   │
│         (collected via all-gather of E integers/rank)     │
│ h: activation of input tokens to MoE block                │
│ E: number of experts                                      │
└──────────────────────────────────────────────────────────┘

┌── Step 1: Compute per-expert per-replica token quota ────┐
│ Parallel for e ← 0 to E:                                  │
│     r_e = Σ_j R_{e,j}  // total replicas for expert e    │
│     t_e = Σ_j T_{e,j}  // total tokens routed to e       │
│     p_e = t_e / r_e    // tokens each replica handles    │
│                                                           │
│ Parallel for j ← 0 to N:                                  │
│     P_{e,j} = p_e × R_{e,j}  // rank j capacity for e   │
│     P_{e,j} = P_{e,j} - min(P_{e,j}, T_{e,j})           │
│     // subtract tokens already local to j                │
│     // remaining P_{e,j} = residual capacity             │
│                                                           │
│ D_{e,i} = p_e × R_{e,i} - P_{e,i}  // locally processed │
└──────────────────────────────────────────────────────────┘

┌── Step 2: Distribute overflow tokens ────────────────────┐
│ Parallel for j ← 0 to N, j ≠ i:                           │
│     D_{e,j} = (T_{e,i} - D_{e,i}) ×                      │
│               P_{e,j} / Σ_{k≠j} P_{e,k}                 │
│     // distribute rank i's remaining tokens              │
│     // proportionally to other ranks' residual capacity  │
└──────────────────────────────────────────────────────────┘

┌── Step 3: Compute dispatch counts per rank ──────────────┐
│ Parallel for j ← 0 to N:                                  │
│     s_j = Σ_e D_{e,j}  // total tokens to rank j        │
└──────────────────────────────────────────────────────────┘

┌── Step 4: Reshuffle activations ─────────────────────────┐
│ Parallel for j ← 0 to N:                                  │
│     Parallel for e ← 0 to E:                              │
│         start = Σ_{0..j-1} s_{j'} + Σ_{0..e-1} D_{e',j}│
│         end = Σ_{0..j-1} s_{j'} + Σ_{0..e} D_{e',j}    │
│         h'[start..end] = tokens in h routed to e that   │
│            are dispatched to rank j, starting from the   │
│            (Σ_{j'=0}^{j-1} D_{e,j'})-th token           │
│                                                           │
│ // h' is now sorted by (target_rank, expert_id)          │
└──────────────────────────────────────────────────────────┘

┌── Step 5: Flexible All-to-All ───────────────────────────┐
│ // Unlike vanilla EP's padded all-to-all                  │
│ // each rank j receives exactly s_j tokens                │
│ // (no padding, no wasted communication)                  │
│ Dispatch all-to-all: h' → remote ranks (s_j tokens each) │
│ Expert computation on received tokens                     │
│ Combine all-to-all: results → original ranks              │
└──────────────────────────────────────────────────────────┘
```

  - **MRO Expert Placement 执行原理全过程**：
```
┌── Input ────────────────────────────────────────────────┐
│ E: experts, sorted by popularity (ascending)             │
│ r_e: replica count for each expert e                     │
│ N: number of nodes                                       │
│ c: replica slots per node                                │
└──────────────────────────────────────────────────────────┘

┌── Case 1: E ≤ c (simple case) ─────────────────────────┐
│ Strategy:                                                │
│ - First r_1 nodes: place experts {1, 2, ..., E}         │
│ - First r_2 nodes: place experts {2, ..., E}            │
│ - ...                                                    │
│ - First r_E nodes: place expert {E}                     │
│ Result: S_1 ⊂ S_2 ⊂ ... ⊂ S_E                          │
│ Recovery probability = P(any of first r_1 nodes alive) │
│ This is the theoretical upper bound → optimal            │
└──────────────────────────────────────────────────────────┘

┌── Case 2: E > c (difficult case) ──────────────────────┐
│ Step 1: Partition experts into ⌈E/c⌉ groups:            │
│         Group 1: experts {1, ..., c}                     │
│         Group 2: experts {c+1, ..., 2c}                  │
│         ...                                              │
│                                                          │
│ Step 2: Partition first ~nodes:                          │
│         Group 1 gets r_1 nodes                           │
│         Group 2 gets r_{c+1} nodes                       │
│         ...                                              │
│                                                          │
│ Step 3: For each group i, place experts                  │
│         in group i on its assigned nodes                 │
│         using the simple case strategy                   │
│         → S_{lo} ⊂ ... ⊂ S_{hi} within group           │
│                                                          │
│ Step 4: Fill vacant slots with remaining replicas       │
│                                                          │
│ Result: Recovery requires ≥1 node alive in each         │
│         group's representative expert set               │
│ Optimality: Theorem 1 proves MRO maximizes              │
│         Pr(∪_{a∈A} Col_a = [E]) for given r_e          │
└──────────────────────────────────────────────────────────┘
```

  - **Reconfiguration 流程**：
```
Failure detected by Controller (heartbeat timeout)
  ↓
Controller re-computes expert allocation + MRO placement
  using only remaining alive nodes (<100ms)
  ↓
Greedy node mapping: min(#new experts per node)
  physical_node → placement_plan_node
  ↓
Agent relays new plan → Lazarus runtime
  ↓
NCCL enqueued ops timeout (10~20s)
  ↓
NCCL communication groups reconfigured (5~15s)
  ↓
State transfer: batched NCCL send/recv from owning nodes
  (e.g. GPT-L: 160 expert states, 7.6s transfer time)
  ↓
Training resumes with all remaining GPUs fully utilized
Total reconfiguration time: 20~40s
```

  - **关键性能数据**：
    | 场景 | GPT-S (521M, 8E) | GPT-L (1.7B, 16E) |
    |------|-------------------|--------------------|
    | 5min MTBF vs DS | 2.8× | 5.7× |
    | 40min MTBF vs DS | 1.6× | 2.3× |
    | 5min MTBF vs DS(FT) | 1.4× | 2.8× |
    | Spot trace vs DS | 2.3× | 3.4× |
    | Spot trace vs DS(FT) | 1.2× | 1.8× |
    | Laz. throughput (no failure, GPT-M) | 45 samples/s | DS: 34 samples/s |
    | 4-node failure recovery prob (GPT-L) | 41% (Lazarus) | 12% (spread) |
