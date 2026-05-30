## MoE Gating Kernel Fusion (MoE 门控 Kernel 融合)

术语解释
MoE Gating Kernel Fusion 是 DeepSpeed-MoE 推理系统提出的 CUDA kernel 优化：将 Gating 函数的 Top-K 选择、Cumsum（Blelloch scan 并行前缀和）、Scatter 等操作融合为单个 CUDA kernel，使用 dense token-to-expert mapping table 替代 sparse one-hot 表示，消除 sparse einsum 的大量零运算，实现 6x+ MoE kernel 延迟降低。

术语是什么？
传统 MoE 实现中 Gating 相关计算分为多个独立 kernel：创建 sparse one-hot mask（S x E 矩阵，其中 (E-1)/E 为零）、Sparse×Dense Einsum（S x E x M x ce，其中 (E-1)/E 为与零相乘）、Cumsum（计算每 expert 处理多少 token）。这些操作因 sparse 表示导致大量无效计算和 kernel launch overhead。

DeepSpeed-MoE 的关键优化：
1. **Dense Mapping Table**：用 `expert_id[S]`（每 token 目标 expert）和 `local_id[S]`（在同 expert 的 tokens 中的位置）替代 S×E sparse one-hot
2. **Kernel Fusion**：Top-K + Cumsum + Scatter 合并为单个 kernel
3. **Blelloch Scan**：并行前缀和算法实现 GPU 上高效的 Cumsum
4. **Data Layout Transform**：替代 sparse einsum 进行 token 排序/反排序

从kernel调度角度拆解术语：
```
// Fused Gating Kernel (1 CUDA kernel, S threads)
// Input:  gate_logits[S][E],  S=num_tokens, E=num_experts
// Output: expert_id[S], local_id[S], expert_offset[E+1]

__global__ void fused_moe_gate(
    float* gate_logits, int S, int E,
    int* expert_id, int* local_id, int* expert_offset)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= S) return;
    
    // Phase 1: Top-1 selection
    float max_logit = -INFINITY;
    int best_expert = 0;
    for (int e = 0; e < E; e++) {
        if (gate_logits[tid * E + e] > max_logit) {
            max_logit = gate_logits[tid * E + e];
            best_expert = e;
        }
    }
    expert_id[tid] = best_expert;
    
    // Phase 2: Atomic count per expert
    int pos = atomicAdd(&expert_offset[best_expert + 1], 1);
    local_id[tid] = pos;
}

// After kernel: Blelloch Scan on expert_offset to compute prefix sum
// expert_offset[i] = sum_{j=0}^{i-1} expert_counts[j]
// Gives starting position for each expert's tokens in output buffer

// Data Layout Transform (替代 Sparse Einsum):
// From: O = Softmax(G) ⊙ X  (sparse-dense einsum, S×E×M×ce ops)
// To:   for t in 0..S: output[expert_offset[expert_id[t]] + local_id[t]] = input[t]
// Complexity: O(S×E×M×ce) → O(S×M×ce)
```

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理系统（开源：https://github.com/microsoft/DeepSpeed）
- Blelloch Scan 是 GPU 上高效的并行前缀和算法：up-sweep (reduce) + down-sweep (propagate)
- 数据布局变换融合 gating probability 缩放：在反排序时直接乘以对应的 gate probability
- 组合优化实现 MoE kernel 延迟降低 6x+

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
