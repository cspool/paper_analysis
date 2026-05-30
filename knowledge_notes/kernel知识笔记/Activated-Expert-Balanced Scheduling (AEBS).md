## Activated-Expert-Balanced Scheduling (AEBS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AEBS 是 JANUS 提出的 MoE 层激活 expert 调度算法，实现为 GPU kernel，在每 MoE 层每 decode step 运行。其核心思想是：MoE 层 latency 由所有 MoE instance 中 distinct activated expert 数最多的那个 instance（即 a_max = max_i a_i）决定。因此，调度目标不是平衡 token counts 或 routing probabilities（如 EPLB），而是直接 minimize a_max。

AEBS 是 synchronization-free 的——每个 MoE instance 独立运行相同的 deterministic kernel，通过确定性算法保证所有 instance 产生相同的调度决策，无需跨 GPU 协调或 CPU-GPU 同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Algorithm 1: AEBS (CUDA Kernel, per MoE layer, per decode step)

Input (GPU global memory):
  T: token count, k: top-k, n_e: MoE instance count
  L(i,j): logical expert ID for token i, expert j    [T × k]
  R(e): replica count for expert e                    [E]
  G(e): set of instances hosting replicas of expert e [E]
  P(e,g): physical replica ID of expert e on instance g [E × n_e]

// Step 1: Collect activated expert set (GPU parallel)
Parallel for (i in 0..T-1, j in 0..k-1):
    atomicOr(E_bitmap, L(i,j))  // bit vector marking activated experts

// Step 2: Initialize per-instance load counters
load[g] = 0 for g = 1..n_e   // distinct expert count per instance

// Step 3: Assign single-replica experts (forced placement)
for e in E_active where R(e) == 1:
    g = unique_instance(G(e))
    actRep[e] = P(e,g)
    atomicAdd(load[g], 1)

// Step 4: Assign multi-replica experts (greedy load balancing)
for e in E_active where R(e) > 1:
    g* = argmin_{g ∈ G(e)} load[g]  // instance with fewest activated experts
    actRep[e] = P(e, g*)
    atomicAdd(load[g*], 1)

// Step 5: Rewrite token routing (GPU parallel)
Parallel for (i in 0..T-1, j in 0..k-1):
    O(i,j) = actRep[L(i,j)]  // logical EID → physical RID

// Step 6: Dispatch (performed by each MoE instance independently)
// Each instance reads O to determine which tokens to process locally

Key invariants:
  - ALL n_e instances run identical kernel with identical input
  - AEBS is deterministic → same output on all instances
  - No cross-GPU communication needed for scheduling
  - No CPU-GPU synchronization (pure GPU kernel)
```

Performance characteristics:
- Scheduling overhead: <20μs (batch=64) to <90μs (batch=4096), plateaus when most experts activated
- AEBS vs EPLB: reduces a_max by 2-5 experts → MoE layer latency reduction
- Scales well with MoE instances (8→16: only small overhead increase)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现为 CUDA kernel (~300 行 CUDA/C++)，作为 SGLang MoE layer 的一部分
- Input data (top-k routing results) 已在 GPU global memory (gating kernel 输出)，无需 CPU 访问
- Replica mapping metadata 更新频率低 (仅在 reconfiguration 时，~15min 间隔)，可放入 GPU constant memory
- 所有 MoE instances 使用相同 input 独立运行（通过 NVSHMEM broadcast 或共享的 routing data 保证一致性）
- 适用于任何有 expert replica 冗余的分布式 MoE 推理/训练系统

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
