## All-gather Dispatch for Expert Parallelism

术语解释
METRO 提出的 Expert Parallelism 通信模式变体，将 MoE layer dispatch 阶段的 all-to-all 通信替换为 all-gather，使每个 GPU 获得全局 token 集合，从而能在每个 GPU 上独立计算全局 top-k 和 token routing 决策。这解决了传统 all-to-all dispatch 下各 GPU 仅有本地 top-k 信息、无法做出全局最优 routing 决策的问题。

术语是什么？
传统 EP dispatch 流程：每个 GPU 对本地 tokens 计算 top-k → all-to-all 将 tokens 发送到对应 expert 所在 GPU。各 GPU 仅知本地 top-k，不知其他 GPU 的 expert 选择情况。METRO all-gather dispatch：每个 GPU 将本地 tokens all-gather 到所有 GPU → 每个 GPU 现在持有全局所有 tokens → 在全局 token 集上计算 top-k → 每个 GPU 独立获得完整的全局 T[1..N]（每个 expert 的全局 token 计数）→ 执行 METRO routing algorithm → 仅计算分配给本 GPU 的 experts → all-to-all combine。

从kernel调度角度拆解术语：
All-gather dispatch 的 kernel 级执行序列：

```
=== METRO All-gather Dispatch（8 GPUs, 32 tokens/GPU）===

// 原 all-to-all dispatch:
// Step A1: 各 GPU local top-k → T_local[1..N]
// Step A2: All-to-all (tokens dispatch) → 每个 GPU 收到目标 expert 的 tokens
// Step A3: FFN compute
// Step A4: All-to-all (results combine)
// 问题: 各 GPU 只有 T_local[1..N]，无法做全局 routing 优化

// METRO all-gather dispatch:
// Step M1: All-gather tokens
//   数据量: 32 tokens/GPU × 8 GPUs × hidden_dim × fp16
//          = 32 × 8 × 1536 × 2 bytes ≈ 768KB → 每 GPU 收到 6MB (after all-gather)
//   NCCL kernel: all-gather 在 NVLink 上 ~3μs bandwidth + ~100μs launch
//   对比: all-to-all 256KB/GPU → ~400ns bandwidth + ~100μs launch
//   差异: bandwidth 增加 ~2.6μs，远低于 NCCL launch overhead

// Step M2: Global Top-K (CUDA kernel)
//   for each GPU g (并行):
//     对全局 256 tokens 计算 router logits
//     for each token t:
//       gate_logits = h[t] @ W_gate
//       probs = softmax(gate_logits)
//       top_k = TopK(probs, K)
//       更新 T[1..N] (每个 expert 的全局 token 计数)
//   开销: 17→20μs (local) vs 原 17→19μs, +3μs max

// Step M3: METRO Routing (CUDA kernel, 单 SM)
//   执行 Algorithm 1: greedy expert-to-GPU assignment
//   见 METRO greedy algorithm entry
//   开销: 17→26μs

// Step M4: FFN Compute
//   仅计算分配给本 GPU 的 activated experts
//   开销: 230→311μs (varies with replication, ~81μs 减少 vs EPLB)

// Step M5: All-to-all Combine
//   同原流程，将 expert outputs 返回各 token 原 GPU
//   开销: 与 EPLB 基线相同
```

术语一般如何实现？如何使用？
- All-gather dispatch 是 METRO 的关键使能技术——没有全局 top-k 信息 T[1..N]，就无法做出最小化 activated experts 的 routing 决策
- 开销分析：在 memory-bound decode 小 batch 下，NVLink latency 主导通信开销（~100μs NCCL launch），bandwidth 差异（all-gather 2MB/GPU vs all-to-all 256KB/GPU = ~2.7μs on 600 GB/s NVLink）可忽略
- 适用条件：decode phase（memory-bound, small batch）；prefill phase（compute-bound, large batch）继续使用 all-to-all + EPLB token routing，因为 prefill 下 bandwidth 开销会放大
- 冗余 top-k 计算：all-gather 后每个 GPU 在全局 token 集上计算 top-k，产生冗余计算（8 GPUs 各算一次），但 top-k 计算量极低（<5% layer time），冗余带来的 overhead <1%

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---
