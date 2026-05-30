## Expert Parallelism with Node-Limited Routing (专家并行与节点限制路由)

术语解释
Expert Parallelism (EP) 是 MoE 模型的分布式并行策略：将不同 expert 的完整权重分配到不同 GPU，token 通过 all-to-all 通信被发送到持有其选中 expert 的 GPU 进行计算。DeepSeek-V3 使用 64-way EP（跨 8 nodes）并引入 Node-Limited Routing 约束（每 token 最多路由到 M=4 个节点），使 IB 跨节点流量可控，配合 DualPipe 和 warp specialization 通信 kernel 实现近乎零通信开销。

术语是什么？
DeepSeek-V3 EP 配置：64 GPUs 承载 256 routed experts + 1 shared expert × 58 MoE layers。每个 GPU 承载 4 个 experts（256/64=4）。训练时使用 16-way PP + 64-way EP + ZeRO-1 DP。Node-Limited Routing：每 token 最多路由到 M=4 个节点，平均每节点选 3.2 experts，实际 K_r=8（理论上可扩展到 13 而通信量不变）。推理时 prefill 使用 EP32，decode 使用 EP320。

从kernel调度角度拆解术语：
```
=== DeepSeek-V3 EP with Node-Limited Routing ===

// Training configuration: 64 GPUs (8 nodes × 8 GPUs/node)
// PP=16, EP=64, ZeRO-1 DP

// Node-Limited Gating (per token)
selected = TopK({s_{j,t} + b_j | j=1..256}, K_r=8)
// Check node constraint:
nodes_used = {node_of(expert_j) for j in selected}
if len(nodes_used) > 4:  // M=4
    // Re-select TopK but limit to at most 4 nodes
    // Each node keeps its top (K_r/M=2) experts by affinity
    for node in nodes_used (keep top 4 by total affinity):
        keep top 2 experts per node

// All-to-All Dispatch flow (warp specialization, 20 SMs)
for token t:
    for expert e in selected[t]:
        target_gpu = expert_to_gpu[e]
        if node_of(target_gpu) != node_of(current_gpu):
            // Cross-node: IB → NVLink pipeline
            IB_send(activation[t], node_of(target_gpu),
                    gpu_idx_within_node(target_gpu))
            // On target node: NVLink forward to expert GPU
        else:
            // Intra-node: NVLink only
            NVLink_send(activation[t], target_gpu)

// Expert FFN Execution
for gpu in 0..63:
    process_batched_tokens_for_4_local_experts()

// All-to-All Combine
// Reverse of dispatch, with FP32 accumulation at aggregation points
```

术语一般如何实现？如何使用？
EP vs TP trade-off：TP 将每个 expert 权重切分到多 GPU，每 GPU 参与每 token 计算但通信开销大；EP 每 GPU 仅处理路由到本地的 token，通信量与激活 expert 数成正比但 GPU 负载可能不均。Node-Limited Routing 是 EP 的关键优化：限制跨节点通信量，配合 IB/NVLink 带宽差异（NVLink 160 GB/s vs IB 50 GB/s ≈ 3.2×）。DeepSeek-V3 的 EP 实现依赖自研 HAI-LLM 框架，而非标准 NCCL all-to-all。NCCL all-to-all 不做 node-limited 优化，浪费 IB 带宽。

涉及论文标题：
- DeepSeek-V3 Technical Report
