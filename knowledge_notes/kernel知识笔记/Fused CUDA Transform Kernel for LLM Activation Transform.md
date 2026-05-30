## Fused CUDA Transform Kernel for LLM Activation Transform

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused CUDA Transform Kernel 是 ParoQuant 为 scaled pairwise rotation 逆变换 T^{-1}(X) 实现的单一融合 CUDA kernel。该 kernel 将 channel-wise scaling 逆变换（X·diag(1/α)）和 K 个 independent Givens 旋转的逆变换（R_1^{-1}·...·R_K^{-1}）融合为单次 kernel 调用。通过三次并行策略实现高效执行：(1) Token 级并行——沿 batch×seq_len 维度分配 grid stride loops；(2) Channel Group 级并行——不同 CUDA block 处理不同 channel group（group_size=128）；(3) Pair 级并行——同一 group 内不同 CUDA thread 处理不同 Givens 旋转对。由于 group size 小（128 × FP16 = 256 bytes），激活 tile 可完全放入 on-chip shared memory，旋转参数（pair indices + angles）存入寄存器。所有 pair 间无数据依赖（synchronization-free），8 个 rotations 在一个 kernel 内依次应用，无需多次 global memory 往返。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Kernel 三级并行执行流程：
```
# CUDA kernel 伪代码
__global__ void scaled_pairwise_rotation_inverse(
    half* X,           // [batch, seq_len, C_in] FP16 激活
    half* X_out,       // 输出
    float* alpha,      // per-channel scaling [C_in]
    int* pair_indices, // [K, max_pairs_per_group, 2]
    float* angles,     // [K, max_pairs_per_group]
    int C_in, int K, int group_size
) {
    // Level 1: Token-level grid stride
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_offset = tid * C_in;  // 每个 token 的起始位置
    
    // Level 2: Group-level (shared memory)
    __shared__ half X_shared[128];  // 256 bytes, 远小于 typical 48KB
    
    for (int g = 0; g < C_in / group_size; g++) {
        int g_start = token_offset + g * group_size;
        
        // Load group tile from global to shared memory (coalesced)
        X_shared[threadIdx.x] = X[g_start + threadIdx.x];
        __syncthreads();
        
        // Level 3: Pair-level (register, synchronization-free)
        // 先做 inverse scaling: X *= 1/alpha
        X_shared[threadIdx.x] *= (half)(1.0f / alpha[g*group_size + threadIdx.x]);
        
        // K independent rotations 依次应用
        for (int k = 0; k < K; k++) {
            if (threadIdx.x < num_pairs_per_group) {
                int i = pair_indices[k][threadIdx.x][0];
                int j = pair_indices[k][threadIdx.x][1];
                float c = cosf(angles[k][threadIdx.x]);
                float s = sinf(angles[k][threadIdx.x]);
                // Inverse: X' = X * G(i,j,-theta), i.e. angle = -theta
                // cos(-θ)=cosθ, sin(-θ)=-sinθ
                half xi = c * X_shared[i] + s * X_shared[j];
                half xj = -s * X_shared[i] + c * X_shared[j];
                X_shared[i] = xi;
                X_shared[j] = xj;
            }
        }
        __syncthreads();  // 仅需在 rotations 间同步
        
        // Write back to global
        X_out[g_start + threadIdx.x] = X_shared[threadIdx.x];
    }
}
```
关键设计：(1) 同一 rotation 内所有 pairs 无同步——各 thread 独立读写不同的 shared memory 位置，无数据竞争；(2) Rotation 间需一次 __syncthreads() 但不需 global memory 往返；(3) Shared memory 仅 256 bytes/group，远小于 48-100KB SRAM，可实现高 occupancy。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 kernel 在 ParoQuant 推理 pipeline 中的位置：对每个 W4A16 量化 linear 层，先调用 transform kernel 对激活 X 应用 T^{-1}，再调用 AWQ W4A16 GEMM kernel 执行 INT4 矩阵乘法。相比 Hadamard transform：(1) ParoQuant kernel 的 channel 维度越大加速比越显著（Figure 4）——因为 Hadamard 有全局依赖需 O(C log C) 步，而 ParoQuant 的组内并行与 C 无关；(2) 组级独立性使各 CUDA block 负载均衡——每个 block 处理固定 128 通道，不随 C 增长而变。在 RTX A6000 上，ParoQuant 端到端 decode 吞吐仅比 AWQ（无 transform）慢约 10%（如 Qwen3-4B: 160 vs 176 tokens/s），比 QTIP（Hadamard transform）快约 25%（160 vs 117 tokens/s）。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
