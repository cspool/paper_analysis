## Selective Scan (Parallel Scan in Mamba/Mamba-2)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Selective Scan是Mamba/Mamba-2中实现SSM状态递归更新的核心计算。在标准RNN中，隐状态更新h_t = A_t * h_{t-1} + B_t * x_t是严格串行的。Mamba利用该递归的线性性质，将其视为binary associative operator作用于有序序列上的prefix sum问题，通过parallel scan（parallel prefix sum的推广）算法将O(L)的串行时间复杂度降低为O(log L)并行步骤。Mamba的selective scan因A_t依赖输入（非时不变），需使用更复杂的associative scan。Mamba-2则利用SSD对偶性将scan重构为分块矩阵乘法（chunked parallelism），chunk内使用高效MatMul，chunk间使用recurrent传递状态，充分利用Tensor Core。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Chunked Parallel Scan（Mamba-2 SSD kernel）伪代码：
```
// Input: A_bar ∈ R^{B×L×H×P}, B_bar ∈ R^{B×L×H×P}, C ∈ R^{B×L×H×P}, V ∈ R^{B×L×H×P}
// H=head数, P=head_dim, L=seq_len
// chunk_size = min(64, L)

num_chunks = ceil(L / chunk_size)

// Phase 1: Intra-chunk (parallel across all chunks via MatMul)
for each chunk i in parallel:
    // Chunk内SSM scan, 等价于:
    // M_chunk = L_chunk ◦ (C_chunk @ B_chunk^T)  // 半可分离矩阵
    // Y_chunk = M_chunk @ V_chunk  // MatMul, 利用Tensor Core
    Y_local[i], h_final[i] = chunk_ssd(Q_chunk, K_chunk, V_chunk, h_init=0)

// Phase 2: Inter-chunk (recurrent, sequential)
h_running = 0
for i in 0..num_chunks-1:
    // 用前一chunk的最终状态调整当前chunk输出
    Y[i] = Y_local[i] + correction_term(h_running, C_chunk, V_chunk)
    h_running = h_final[i] * exp(-Δ_sum) + correction_state(h_running, ...)
```

Mamba-1的selective scan无chunked策略，纯parallel scan比Mamba-2慢约8x。

术语一般如何实现？如何使用？
Mamba开源实现提供CUDA kernel（https://github.com/state-spaces/mamba）。Mamba-2实现使用Triton kernel（chunked scan）。论文使用Megatron-LM中的实现。核心优化原则：对长序列使用chunked策略并行化；保证memory coalescing访问；避免thread divergence。在GPU上，Mamba-2的SSD scan比Mamba-1 scan快8倍。

SAMBA 论文使用 Mamba 的硬件感知并行扫描算法实现高效训练，同时在混合架构中与 FlashAttention 2 配合：SWA 层（窗口=2048）使用 FA2——训练速度与 Mamba 的 selective parallel scan 在 seqlen=2048 时相当（基于 Gu & Dao 2023 测量）。推理时 Mamba 层 O(1) 状态更新 + SWA 层 O(window) 计算，总体解码仍为线性时间复杂度。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models
- Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

---
