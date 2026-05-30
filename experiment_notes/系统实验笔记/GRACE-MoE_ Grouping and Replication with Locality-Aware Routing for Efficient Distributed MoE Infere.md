## GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  - GRACE-MoE 在 Megablocks 上构建了一套面向多节点 SMoE 推理的 Serving 调度系统，核心调度/系统优化包括：
    1. **Hierarchical Sparse Communication (HSC, Section 5)**：替换 flat global All-to-All 为 physically global but logically sparse 的两阶段通信方案——Stage 1 跨节点路由（每个 GPU 与远端节点 peer GPU 通信，将 token 转发到目标节点，dest node 相同的 token 仅传输一次）；Stage 2 节点内重分发（GPU 间 token 经 NVLink 传输到 expert 所在 GPU）。跨节点使用单一 global communication group + zero-padding 实现 logically sparse point-to-point transfers，保留 sparse communication 的带宽优势并利用 global collective 的 implicit barrier 做 soft synchronization。跨节点通信与节点内 routing decision computation 通过细粒度 pipelining 重叠。
    2. **Offline-Online Coordinated Scheduling**：Offline 阶段（profiling → grouping → replication）生成 expert placement plan 和 replica map；Online 阶段（HSC + topology-aware routing）按 plan 执行 token dispatch 和 replica selection。Offline 结果可跨 dataset 复用（cross-dataset placement 最差 latency 增加 ≤4.52%），避免频繁 re-profiling。
    3. **Multi-Node Multi-GPU Synchronization Reduction**：通过 HSC 的 implicit barrier 机制消除 explicit global barrier，结合 locality-aware routing 优先使用本地/节点内副本减少跨节点通信，缓解 straggler effect 和 synchronization overhead。
  - 实验比较：(1) 端到端 inference latency 和 MoE layer time：GRACE-MoE vs Tutel, Megablocks, vLLM, C2R, Occult；(2) 六种组件增量配置的通信/负载/延迟指标分解（Table 1, Figure 5）；(3) Cross-dataset transfer generalizability；(4) Lighter workloads（batch=64/128）下的稳定性。

- 硬件平台是什么，配置是什么。
  - 2 节点，每节点 4× NVIDIA A100-SXM4 GPU (80GB)。节点内 NVLink（12 links/GPU, 50 GB/s per direction）。节点间 25 Gbps Ethernet（模拟实际有限跨节点带宽）。
  - 软件：Megablocks (Gale et al. 2023) + PyTorch 2.5 + Triton 3.1，支持 data parallelism + expert parallelism。

- 开源Serving框架是什么。修改了什么。
  - 基于 Megablocks（https://github.com/databricks/megablocks），一个基于 block-sparse matrix multiplication 的 MoE 计算框架。
  - GRACE-MoE 修改/新增内容：
    - **HSC 通信模块**：替换 Megablocks 原有的 flat All-to-All 为 hierarchical sparse communication 实现。Cross-node 部分使用 global collective group + zero-padding sparse point-to-point；Intra-node 使用节点内高带宽链路做 token redistribution。Cross-node 通信与 intra-node routing computation fine-grained pipelining。
    - **Offline Profiling + Grouping 模块**：基于 spectral clustering 实现 hierarchical grouping（跨节点 fully non-uniform + 节点内 controlled non-uniform），生成 expert placement map。
    - **Dynamic Replication 模块**：基于 load skew factor ρ 计算每层 replica 数，选择 heaviest group 中 hot experts 复制到 underutilized GPUs。
    - **Online Routing 模块**：Topology-aware routing with locality preference（三级优先 + WRR with load prediction），集成到 Megablocks 的 token dispatch 路径中。
    - **数据并行 + Expert 并行**：保持 Megablocks 原有 DP+EP 能力，在其上构建 HSC + routing。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 论文声明 "code will be released upon acceptance"，截至分析时未发现公开仓库。基于开源 Megablocks。GRACE-MoE 作为 Megablocks 的上层调度优化，不修改底层 expert kernel。
  - Serving 框架执行全过程（以 OLMoE 6.92B, 2 nodes×4 GPUs/node, batch=512, prefill=64, decode=32 为例）：

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Offline 准备阶段                                             │
│    - 在 calibration data (WikiText-2) 上 profiling              │
│    - 记录每层每个 expert 的共激活频率 → affinity matrix A[l]    │
│    - Hierarchical Grouping:                                     │
│      Cross-node (fully non-uniform): 64 experts→2 node groups   │
│      Intra-node (controlled non-uniform, r=0.15): 每组→4 GPU groups│
│    - Dynamic Replication: 每层根据 ρ 确定副本数和 hot experts   │
│    - 生成 expert placement plan + replica map                    │
│           ↓                                                      │
│ 2. 模型加载                                                     │
│    - Megablocks 加载 MoE 模型权重到各 GPU                        │
│    - 按 placement plan 分布 expert，按 replica map 复制 hot     │
│      expert 权重到 secondary GPUs                               │
│    - Attention/Embedding/Norm 常驻各 GPU (DP 复制)               │
│           ↓                                                      │
│ 3. 用户输入 batch tokens [B=512, S=64]                          │
│    Prefill + Decode loop:                                        │
│           ↓                                                      │
│ 4. 每层 MoE 执行（以单层为例）                                   │
│    ┌─ Router/Gating ────────────────────────────────────────┐   │
│    │  gate_logits = W_gate @ h  [512×64, 64 experts]        │   │
│    │  topk_indices, topk_weights = topk(softmax(logits), k) │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ HSC Stage 1: Cross-node Token Forwarding ─────────────┐   │
│    │  每个 GPU 扫描所有 expert indices                        │   │
│    │  对 dest node 相同的 token：聚合为单次 cross-node send  │   │
│    │  Global collective group + zero-padding → sparse P2P    │   │
│    │  跨节点 traffic 仅含必要的 token data（去重后）          │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ HSC Stage 2: Intra-node Redistribution (与 Stage 1     │   │
│    │  routing decision computation pipelined) ───────────────│   │
│    │  节点内各 GPU 经 NVLink (50GB/s×12) P2P 传输 tokens     │   │
│    │  Token 精准路由到 expert 所在 GPU                        │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Locality-Aware Routing (on replica selection) ────────┐   │
│    │  for each token → expert e with replicas:               │   │
│    │    if token GPU has replica: use local (intra-GPU)      │   │
│    │    elif intra-node replica exists: WRR w/ load pred     │   │
│    │    else: cross-node WRR fallback                        │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ Expert FFN Computation ───────────────────────────────┐   │
│    │  Megablocks block-sparse matmul:                        │   │
│    │  gate_out = SiLU(W_gate @ x) * (W_up @ x)              │   │
│    │  expert_out = gate_out @ W_down                        │   │
│    │  output = sum(topk_weights * expert_out)                │   │
│    └────────────────────────────────────────────────────────┘   │
│    ┌─ HSC Combine (对称反向) ───────────────────────────────┐   │
│    │  Intra-node gather → cross-node combine (global group)  │   │
│    │  Token output 返回到原 GPU，reassemble 序列              │   │
│    └────────────────────────────────────────────────────────┘   │
│           ↓                                                      │
│ 5. 输出: generated tokens                                       │
│    End-to-end latency 相比 Occult 降低 up to 78.55%              │
│    4.66× speedup (OLMoE, 2 nodes×4 GPUs)                        │
└─────────────────────────────────────────────────────────────────┘
```

  - HSC 的关键设计优势：flat global All-to-All 需要 strict synchronization across all ranks，heterogeneous 集群中受最慢链路限制（straggler effect）。HSC 通过 global collective 的 implicit barrier 做 soft sync + logically sparse transfer，消除 explicit barrier 开销。Cross-node traffic 通过 token deduplication（同一 dest node 多 token 聚合单次传输）进一步减少。
  - Component analysis (Table 1, vs Occult baseline):
    - Occult: uniform grouping + flat All-to-All
    - Occult+HSC: All-to-All time −35.19%, GPU idle −49.88%
    - HG+HSC: All-to-All time −48.33%, cross-node traffic −50.67%, GPU load std +90.03%
    - +DR+WRR: GPU idle −26.86%, GPU load std +31.92%
    - +DR+TAR (full GRACE-MoE): All-to-All time −50.57%, cross-node traffic −52.11%, GPU idle −25.66%
