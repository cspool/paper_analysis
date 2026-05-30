## Flexible Token Dispatcher (CUDA Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Flexible Token Dispatcher 是 Lazarus 为 MoE 训练设计的 CUDA kernel，实现非对称 expert placement 下的高效 token dispatch。在传统 EP 中，每 expert 仅有一个 replica，token dispatch 是简单的：将 token 发送到持有该 expert 的 GPU。但 Lazarus 为 popular experts 分配了不同数量的 replicas 在不同 GPU 上（非对称 placement），需要决定每个 token 具体发往哪个持有目标 expert replica 的 GPU，同时平衡各 GPU 负载。

该 kernel 对所有 E 个 experts 和 N 个 ranks 并行计算 dispatch schedule。核心逻辑：(a) 计算每个 expert 的每 replica 应处理的 token 数 p_e = t_e / r_e（负载均衡）；(b) 计算每个 rank 对每个 expert 的处理容量 P_{e,j} = p_e × R_{e,j}；(c) 优先将 rank j 本地已有的 token 分配给自身（min(P_{e,j}, T_{e,j})）；(d) 将超出本地容量的剩余 token 按各 rank 剩余容量比例分发（proportional distribution）；(e) 根据 schedule 将 input activations reshuffle 为连续 buffer，使 routed to same expert + dispatched to same rank 的 token 连续排列，供后续 flexible all-to-all collective 使用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Flexible Token Dispatcher 的 CUDA kernel 执行流程（Algorithm 1）：

```
Input: N GPUs, i (current rank), R_{e,j} (replicas for expert e at rank j),
       T_{e,j} (#tokens routed to expert e at rank j), h (input activations)
Output: h' (reshuffled activations for all-to-all), s_j (tokens to rank j)

// Step 1: All-gather T_{e,j} from all ranks (E integers per rank, negligible)
// Step 2: Compute dispatch schedule (parallel across experts and ranks)
for e ← 0 to E in parallel:
    r_e = Σ_j R_{e,j}              // total replicas for expert e
    t_e = Σ_j T_{e,j}              // total tokens routed to expert e
    p_e = t_e / r_e                // tokens each replica should handle
    
    for j ← 0 to N in parallel:
        P_{e,j} = p_e × R_{e,j}    // rank j's processing capacity for expert e
        P_{e,j} -= min(P_{e,j}, T_{e,j})  // subtract locally processed tokens
    
    D_{e,i} = p_e × R_{e,i} - P_{e,i}  // tokens processed locally by rank i
    
    for j ← 0 to N, j ≠ i in parallel:
        // Distribute remaining tokens proportionally to residual capacity
        D_{e,j} = (T_{e,i} - D_{e,i}) × P_{e,j} / Σ_{k≠j} P_{e,k}

// Step 3: Compute dispatch counts per rank
for j ← 0 to N in parallel:
    s_j = Σ_e D_{e,j}

// Step 4: Reshuffle input activations
for j ← 0 to N in parallel:
    for e ← 0 to E in parallel:
        start = Σ_{0..j-1} s_{j'} + Σ_{0..e-1} D_{e',j}
        end = start + D_{e,j}
        // Copy D_{e,j} tokens of expert e from h to h'[start..end]
        // Tokens are sorted by (target_rank, expert_id)

// Step 5: Perform flexible all-to-all with s_j tokens to each rank j
//         (no padding — each rank sends exactly s_j tokens)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Lazarus 用 ~500 LoC CUDA 实现该 kernel，在 MoE block 的 forward path 中替代传统 DeepSpeed MoE 的 dispatch 逻辑。由于 collective communication operations 需要所有参与 rank 的同步，kernel 执行前需先 all-gather 所有 rank 的 T_{e,j}（E integers per rank，overhead 可忽略），确保所有 rank 有全局一致的 dispatch schedule 信息。该 kernel 使用 shared memory 处理 per-rank 的 per-expert capacity 计算，通过原子操作协调跨 warps 的 token 分配。

在 RTX 3090 (10 emulated nodes) 上，当 workload 完全 balance (1:1 load ratio) 时，flexible dispatcher 引入的 overhead 极小（Lazarus 吞吐几乎等于 DS baseline）。当 load ratio 变为 4:1（imbalanced）时，Lazarus 保持恒定吞吐，而 DS 吞吐急剧下降。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
