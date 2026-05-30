## Automatic Parallel Strategy Analyzer (for MoE Serving)

术语是什么？
Automatic Parallel Strategy Analyzer 是 MixServe 的 offline 阶段核心模块，用于自动推导 MoE 模型推理的最优 (d_TP, d_EP, d_DP) 并行策略。它以模型超参数（hidden dim, num_layers, num_experts, top-k）和网络硬件配置（intra-node/inter-node 带宽和拓扑、计算能力和内存）为输入，通过 calibrated theoretical model（计算延迟 τ、通信延迟 λ、排队延迟 Wq 的理论公式）和 profiling data 的联合分析，在满足 NPU 内存约束的前提下，自动选择最小化服务延迟或最大化吞吐量的并行策略。

从系统架构角度拆解术语：
Analyzer 的核心公式和流程：

```
Input:
  - Model: Ψ (weights), h (hidden dim), l (layers), K (top-k)
  - Cluster: n_node, n_proc, BW_intra, BW_inter, M (memory per NPU)
  - Profiling data at various (b, s, d) combos

Computational Latency (式4):
  τ = Ψ / (d_TP · d_EP) · b/d_DP · s · h

Communication Latency (式5):
  λ = 2 × AR(b/d_DP · s·h, d_TP) + 2 × A2A(b/d_DP · s·h·k, d_EP)
  当 d_DP < d_EP 时修正为 A2A(b/d_EP · s·h·k, d_DP)

Service Latency per Token (式6):
  Δt_svc = l[τ + λ] + (d_PP-1) · P2P(b/d_DP · s·h)

Queuing Delay (式7, M/M/1):
  Wq = λ_a / (μ(μ-λ_a)), μ = 1/Δt_svc

Performance Indicators (式9-11):
  TTFT = Wq + Δt_svc|_{s=Lin}
  ITL = Δt_svc|_{s=1}
  Throughput = (Lin+Lout) / (Wq + TTFT + Lout·ITL)

Memory Constraint (式8):
  Ψ_Attn/d_TP + Ψ_MoE/(d_EP·d_TP) + 2·b·s·h · l/d_PP < M
```

枚举满足 n_proc × n_node = d_TP × d_EP 的所有组合 → 在内存约束下选 min TTFT 或 max Throughput。

术语一般如何实现？如何使用？
- MixServe 中 Analyzer 在 offline 运行一次，输出策略供 online Weight Loader + Partitioner 使用。
- 当集群配置变化时重新运行 Analyzer 自动输出新策略。
- 消融实验（Fig. 11）验证了不同硬件平台下 Analyzer 会选择不同的最优配置。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

---
