## Fused GPU Kernel for MoE Coreset Selection

术语解释
一种自定义 CUDA fused GPU kernel，将 MoE dynamic expert sharing 中 coreset selection 的 12 个碎片化算子（softmax、top-k、masked reduction、final ranking 等）融合为仅 2 个 kernel，消除 kernel launch overhead 和冗余 HBM traffic。

术语是什么？
DES 论文（arXiv 2602.00879, 2026）为 Dynamic Expert Sharing 的 coreset selection 阶段开发了自定义 fused kernel。原始 PyTorch 实现需要 12 个独立 kernel（softmax → per-token top-k → mask → cross-sequence reduction → top-k ranking 等），每个 kernel 都需要从 HBM 读写中间结果。

Fused kernel 设计：
- **Primary kernel**：融合 per-token softmax、Top-K filtering 和 weighted expert vote accumulation。使用 register-level computation 保留中间结果（避免 HBM 往返），通过 atomic instructions（atomicAdd）更新全局 expert saliency scores。
- **Second kernel**：基于 threshold-governed ranking 执行 final expert masking。

在 NVIDIA B200 GPU 上实现 **6× speedup** over PyTorch baseline。

从kernel调度角度拆解术语：
```
// Primary Kernel: Per-token fused softmax + topk + vote accumulation
__global__ void fused_vote_kernel(
    float* router_logits,   // [N, M] input
    float* expert_votes,    // [M] output (initialized to 0)
    int N, int M, int K
) {
    int token_id = blockIdx.x;  // one block per token
    if (token_id >= N) return;
    
    // Stage 1: Per-token softmax (register-level, online algorithm)
    float row[M];
    float max_val = -INFINITY, sum_exp = 0.0f;
    float* input = router_logits + token_id * M;
    
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        row[i] = input[i];
        max_val = fmaxf(max_val, row[i]);
    }
    __syncthreads();
    max_val = warpReduceMax(max_val);  // warp-level reduction
    
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        row[i] = expf(row[i] - max_val);
        sum_exp += row[i];
    }
    __syncthreads();
    sum_exp = warpReduceSum(sum_exp);
    
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        row[i] /= sum_exp;  // normalized softmax
    __syncthreads();
    
    // Stage 2: Local top-K mask (register-level)
    // Find K-th largest via parallel selection
    float local_topk_mask[M] = {0};
    // ... topk threshold selection ...
    
    // Stage 3: Weighted vote accumulation (atomic to global)
    for (int i = threadIdx.x; i < M; i += blockDim.x)
        if (local_topk_mask[i] > 0)
            atomicAdd(&expert_votes[i], row[i]);
}

// Second Kernel: Final ranking and masking
__global__ void expert_masking_kernel(
    float* expert_votes,    // [M]
    int* coreset_indices,   // [M_core]
    int M, int M_core
) {
    // Bitonic sort or radix select on expert_votes
    // Output top M_core expert indices
}
```

PyTorch baseline 的 12 kernel 链（内存视角）：
```
logits [N,M] → softmax kernel → gates [N,M] (HBM R/W)
gates → topk kernel → topk_val[N,K], topk_idx[N,K] (HBM R/W)
topk_val → mask kernel → masked_weights [N,M] (HBM R/W)
masked_weights → sum reduce kernel → votes [M] (HBM R/W)
votes → topk kernel → coreset [M_core] (HBM R/W)
coreset → mask/scatter → ... (subsequent routing kernels)
```

Fused 后的 2 kernel 链：
```
logits [N,M] → Primary fused kernel → votes [M] (1 HBM read of logits, atomic writes)
votes [M] → Second kernel → coreset [M_core] (core logic)
```
关键：register-level computation 消除 logits、gates、masked_weights 等中间张量的 HBM 往返（N×M 次 read/write 仅剩 1 次 read），kernel launch 从 12 次降至 2 次。

术语一般如何实现？如何使用？
- 硬件：在 NVIDIA B200 (Blackwell) 上实现，CUDA 13.1。高吞吐架构对算子碎片化更敏感（更受益于融合）
- 实现技术：warp-level reduction、parallel selection for topk、online softmax、atomicAdd for accumulation
- 性能特征：6× speedup over PyTorch baseline in coreset selection；end-to-end GPU kernel time 间接改善 8.2-14.3%
- 适用场景：任何需要跨 token 聚合 routing information 的 MoE optimization（不仅限于 DES）
- 论文未提供开源代码（arXiv 2602.00879）

涉及论文标题：
- DES: Dynamic Expert Selection for Efficient MoE Inference
