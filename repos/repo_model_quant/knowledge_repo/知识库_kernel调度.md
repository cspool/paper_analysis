# 知识库_kernel调度

## Hadamard Kernel Fusion（Hadamard 变换 + 量化/反量化 CUDA Kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hadamard Kernel Fusion 是 SDP4Bit 中的 CUDA kernel 级优化，将 Walsh-Hadamard Transform (WHT) 与对称线性 (de)quantization 操作融合为单个 GPU kernel，消除中间 global memory traffic。核心设计：(1) 每个 CUDA thread block 处理一个 quantization group（大小需被 H 矩阵大小整除）；(2) 从 global memory 加载数据到 shared memory（1 次读）；(3) 在 shared memory 中执行 32×32 Hadamard transform（仅加减运算，memory-bound 在此大小）；(4) 在 shared memory 中直接计算 per-group scale 并执行量化；(5) 将 packed INT4 输出写回 global memory（1 次写）。融合效果：Hadamard transform 的额外开销降至 < 0.3%（Table 5 显示 w/ and w/o Hadamard 的 (de)quantization throughput 差异仅 301.8 vs 305.6 GB/s at 8MB）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused Hadamard + Quantize CUDA kernel 伪代码：
```cuda
// 输入: grad[N] in global memory (FP32)
// 输出: packed_int4[N/2] + scales[N/G] in global memory
// H=32×32, G=512 (group_size), 每个 block 处理一个 group

__global__ void fused_hadamard_quantize_int4(
    const float* grad, uint8_t* packed_out, float* scales,
    int N, int group_size, int H_size)
{
    int gid = blockIdx.x;  // group id
    int tid = threadIdx.x;
    int base = gid * group_size;

    // 1. 加载到 shared memory (1 次 global read)
    __shared__ float smem[512];  // group_size=512
    for (int i = tid; i < group_size; i += blockDim.x)
        smem[i] = grad[base + i];
    __syncthreads();

    // 2. Hadamard transform in shared memory (32x32 block-wise)
    //    H = H_32 ⊗ I_{(G/32)}  (Kronecker product, 无需额外数据移动)
    #pragma unroll
    for (int b = 0; b < group_size; b += H_size) {
        // Fast Walsh-Hadamard Transform: O(H_size * log H_size) adds/subs
        for (int step = 1; step < H_size; step <<= 1) {
            for (int i = tid; i < H_size; i += blockDim.x) {
                int idx = b + i;
                int pair = idx ^ step;  // butterfly pattern
                if (idx < pair) {
                    float a = smem[idx];
                    float b = smem[pair];
                    smem[idx] = a + b;
                    smem[pair] = a - b;
                }
            }
            __syncthreads();
        }
    }

    // 3. 在 shared memory 中量化 (无额外 global memory traffic)
    // 先由 warp reduce 求 max(|group|)
    float local_max = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x)
        local_max = fmaxf(local_max, fabsf(smem[i]));
    // warp-level reduction for scale
    float s = warp_reduce_max(local_max);
    if (tid == 0) scales[gid] = s;

    // 4. 量化并 packed write (1 次 global write)
    float inv_s = 7.0f / s;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int8_t q = roundf(clamp(smem[i] * inv_s, -7.0f, 7.0f));
        // pack 2 INT4 into 1 uint8
        int idx = base / 2 + i / 2;
        if (i % 2 == 0) packed_out[idx] = (q & 0xF);
        else           packed_out[idx] |= (q << 4);
    }
}
```

融合的关键约束：`group_size % H_size == 0`，确保每个 quantization group 可被整数的 Hadamard blocks 覆盖。SDP4Bit 选择 H=32 因为：(1) 32×32 transform 在 GPU 上是 memory-bound（compute 占比极低），融合后开销可忽略；(2) 32×32 足以有效平滑梯度 outlier（Fig. 6 证实）；(3) 与典型 group_size（128, 512）兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SDP4Bit 的 fused kernel 在 `megatron/core/tensor_parallel/` 下实现（CUDA C++），通过 PyTorch C++ extension 注册为 Python 可调用函数。CUDA kernel 使用 `__shared__` memory 存储中间数据，利用 warp shuffle instructions 进行 reduction。Hadamard transform 使用 in-place 的 butterfly 模式（FWHT），无需额外 memory。融合 kernel 支持 INT4 和 INT8 两种输出精度（分别对应 gradient inter-node 和 intra-node 量化）。使用时在 Megatron-LM 训练循环中替换独立的 Hadamard + quantize + dequantize kernel 调用。论文 Table 4 确认融合效果：grad comm time 从 64.6ms（unfused）降至 45.8ms（fused, -29%），E2E TFLOPs 从 55.2 升至 58.5（+6%）。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training

## Dual Form (TTT)（TTT 对偶形式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual form（对偶形式）是 TTT 层中将内循环梯度更新计算转化为矩阵乘法（matmul）操作的数学重写技术。问题背景：primal form（原始形式）需要显式计算每个 token 的外积梯度 G_t = ∇ℓ(W; x_t) = 2(W x̂_t - y_t) x̂_t^T，得到 d×d 矩阵后逐 token 更新 W。这导致两个效率问题：(1) 外积操作无法充分利用 GPU TensorCores（TensorCores 专门优化 matmul，而非外积）；(2) 每个 d×d 矩阵 G_t 的 I/O 开销远大于 d 维向量 x_t。Dual form 的解决方案：通过数学恒等式将 W_b 和 Z = [z_1,...,z_b] 表达为纯 matmul 操作，避免显式存储 G_t。具体地，W_b = W_0 - 2η(W_0X̂ - Y)X̂^T（一次 matmul），Z = W_0X̄ - 2η(W_0X̂ - Y)mask(X̂^TX̄)（两次 matmul + 上三角 mask）。Dual form 在 TPU 上比 primal form 快 5× 以上。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dual form 在第一个 TTT mini-batch 中的 kernel 执行伪代码：

```
# ===== 输入 =====
# X = [x_1,...,x_b] ∈ R^{d×b}     # mini-batch of b tokens
# W_0 ∈ R^{d×d}                    # initial weight (from previous mini-batch)
# η                                # learning rate
# θ_K, θ_V, θ_Q ∈ R^{d×d}         # projection matrices

# ===== Dual Form Kernel =====
# Step 1: 投影（三个独立 matmul，利用 TensorCore）
X̂ = matmul(θ_K, X)     # training view, R^{d'×b}
Y  = matmul(θ_V, X)     # label view, R^{d'×b}
X̄ = matmul(θ_Q, X)     # test view, R^{d'×b}

# Step 2: 计算 mini-batch 结束时的权重（matmul）
# W_b = W_0 - 2η Σ_t (W_0 x̂_t - y_t) x̂_t^T
#      = W_0 - 2η (W_0 @ X̂ - Y) @ X̂^T
E = matmul(W_0, X̂) - Y           # error matrix, R^{d'×b}
W_b = W_0 - 2η * matmul(E, X̂^T)  # weight update, R^{d×d}

# Step 3: 计算所有输出 token（matmul + mask）
# 引用 Fact 1: V · mask(A^T Q) = [Σ_{s=1}^t a_s^T q_t · v_s]_t
# 设置 A=X̂, Q=X̄, V=E, 得到中间量 Δ
S = matmul(X̂^T, X̄)               # similarity matrix, R^{b×b}
S_masked = upper_triangular_mask(S)  # 保留上三角，下三角置零
Δ = matmul(E, S_masked)           # correction term, R^{d'×b}

# Step 4: 最终输出
Z = matmul(W_0, X̄) - 2η * Δ      # output tokens, R^{d'×b}

# ===== 输出 =====
# W_b: 更新后的权重（传递给下一个 mini-batch）
# Z: 当前 mini-batch 的输出 tokens
```

关键观察：
- 所有核心操作均为 matmul（TensorCore 友好）
- 没有显式的外积操作（G_t = 外积）
- S = X̂^T X̄ 的计算是 O(b² × d')，但由于 b=16 很小，实际开销极低
- mask 操作是 element-wise，开销可忽略

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实际 TTT 实现中：
- **Forward (prefill) 模式**使用 dual form：需要并行处理整个 prompt 的 tokens，dual form 将全操作转为 matmul，最大化吞吐量
- **Generate (decode) 模式**使用 primal form：每次仅生成一个 token，无需批处理的 dual form
- Dual form 在 JAX 实现中通过 XLA 自动融合 matmul + mask 操作
- 论文指出 dual form 对 TTT-MLP 同样适用（附录 A），只是符号更复杂（需要处理多层非线性激活），核心思想不变——通过标准反向传播计算 Σ_t G_t^k，再通过 vjp 计算输出

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Gradient Checkpointing Through Time（时序梯度检查点）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gradient checkpointing through time 是将标准梯度检查点（gradient checkpointing / activation recomputation）技术应用于时间维度的内存优化方法。标准梯度检查点（Chen et al., 2016）在训练深度网络时，选择性地不保存某些层的中间激活，在反向传播时重新计算它们，以计算时间换取内存。TTT 层的内循环会产生 T 个中间隐藏状态 W_1,...,W_T（每个 token 对应一个 d×d 矩阵），直接全部保存会消耗不可接受的内存。使用 mini-batch TTT 和 dual form 后，仅需保存每个 mini-batch 结束时的 W（共 T/b 个，而非 T 个），但仍可能过多。Gradient checkpointing through time 在此之上进一步减少内存：仅保存部分 mini-batch 边界的 W，反向传播时从最近的检查点重新计算内循环的 forward pass。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TTT 训练中的 gradient checkpointing 执行流程：

```
# ===== Forward pass (with checkpointing) =====
W_0 = θ_init
checkpoints = []          # 仅保存部分 W

for m in 0, 1, ..., T/b - 1:    # 每个 mini-batch
    if m % checkpoint_interval == 0:
        checkpoints.append((m, W_{m*b}))   # 保存检查点

    # dual form: 从 W_{m*b} 计算 Z_block 和 W_{(m+1)*b}
    Z_block, W_next = dual_form(X_block, W_{m*b}, η)

    # 不保存中间激活（如 Z_block 计算中的中间 matmul 结果）
    # 仅保存 mini-batch 边界的 W 作为检查点

# ===== Backward pass =====
# 从最后一个检查点开始，重新计算 forward 以获取中间激活
# 标准做法：反向遍历检查点，对每段重新执行 forward + backward
for m in reversed(range(num_checkpoints)):
    # rematerialize: 从检查点重新执行该段的 forward
    # 这次保存所有中间激活用于梯度计算
    # backward: 计算梯度并传播
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实践中：
- JAX 和 PyTorch 都提供内置的梯度检查点 API（`jax.checkpoint` / `torch.utils.checkpoint`）
- 标准梯度检查点是按层应用的（每一层可独立重计算），而 TTT 需要按时间步应用（跨多个 mini-batch）
- 论文未详细说明检查点间隔的配置，但指出这是标准技术的直接应用
- 使用 mini-batch TTT (b=16) 后，T/b 个检查点已经远少于 T 个，在 2k 上下文中为 128 个检查点（可行），在 32k 上下文中为 2000 个（需要进一步减少）

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States

## Fused Dequantization + Matrix Multiplication (Q_MatMul / 融合反量化矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Dequantization + Matrix Multiplication（KIVI 中称为 Q_MatMul）是一种 CUDA kernel 优化技术，将低比特张量的反量化（dequantization）和矩阵乘法（matmul）在 GPU kernel 的 tiling 级别融合执行，避免将反量化后的大尺寸 FP16 中间结果写回 GPU 全局内存（HBM）。标准做法是先完整反量化整个量化张量到 FP16 存入 HBM，再执行 matmul 从 HBM 读取。这种做法对于 KV Cache Quantization 而言是巨大的浪费——2bit 量化的 KV Cache 反量化后膨胀 8×（2bit→16bit），相当于量化节省的内存带宽全被反量化浪费。

KIVI 中 Q_MatMul 在 GPU shared memory 内完成：将 query tile 和对应的 quantized KV cache tile 加载到 SRAM → 即时反量化 tile → 在 SRAM 中直接计算 tile 矩阵乘法 → 只将最终结果写回 HBM。避免了 FP16 中间 KV cache 的 HBM 写入和再次读取，有效利用量化减少的内存带宽。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Q_MatMul kernel 伪代码（CUDA 实现）：

```
// 输入: t_Q ∈ R^{M × K} (FP16), Q_X ∈ int2 (packed), scales ∈ FP16, zeros ∈ FP16
// 输出: Out ∈ R^{M × N} (FP16)

__global__ void Q_MatMul(
    half* t_Q,           // [M, K] query tile in shared memory
    uint8_t* Q_X_packed, // [K/4, N] packed 2bit KV cache
    half* scales,        // [num_groups, N] per-group scale
    half* zeros,         // [num_groups, N] per-group zero-point
    half* Out,           // [M, N] output
    int G                // group size = 32
) {
    // 1. 加载 t_Q tile 到 SRAM (寄存器 + shared memory)
    __shared__ half Q_tile[TM][TK];
    // ...load t_Q into Q_tile...

    // 2. 加载 packed 2bit K tile 和 scale/zero 到 SRAM
    __shared__ half K_deq[TK][TN];   // dequantized K tile

    for each element in tile:
        byte = Q_X_packed[packed_idx];
        // 解包: 每 byte 存 4 个 2bit 值
        for bit_idx in [0, 1, 2, 3]:
            val_2bit = (byte >> (bit_idx * 2)) & 0x03;
            group_id = col / G;
            // 即时反量化: 2bit → FP16
            K_deq[row][col] = val_2bit * scales[group_id] + zeros[group_id];

    // 3. Tile matmul in SRAM (不写回HBM)
    for i in range(K):
        for m in range(TM):
            for n in range(TN):
                Out_tile[m][n] += Q_tile[m][i] * K_deq[i][n];

    // 4. 只将最终结果写回HBM
    // ...store Out_tile to Out...
}
```

KIVI 的完整 mixed-precision attention 流程使用两次 Q_MatMul：
1. `A_g = Q_MatMul(t_Q, Q(X_K_g))` — fused dequant+matmul for grouped key
2. `t_O_g = Q_MatMul(A_g_sm, Q(X_V_g))` — fused dequant+matmul for grouped value

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 使用 CUDA 实现 Q_MatMul。实现参考：(1) 映射 tile 到 CUDA thread block，每 block 处理一个输出 tile；(2) 使用 shared memory 缓冲 input tile 和 dequantized tile；(3) 反量化逻辑内联在 matmul 循环中，避免额外的 shared memory buffer。类似技术被广泛使用：FlashAttention 的 online softmax 融合、vLLM 的 fused kernel、CUTLASS 的 mixed-input GEMM。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---

## Tiled Matrix Multiplication for Mixed Precision (分块混合精度矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tiled Matrix Multiplication（分块矩阵乘法）在 KIVI 中指将 grouped quantized 部分和 residual FP16 部分的矩阵乘法分块独立计算后拼接的策略。KIVI 的 KV Cache 分为两部分：grouped 量化部分 `Q(X_K_g)` (2bit) 和 residual 全精度部分 `X_K_r` (FP16)。Attention score 计算 `A = t_Q X_K^T` 无法直接执行因为两部分精度和布局不同。Tiled matmul 将 X_K 视为两个 tile，分别用不同 kernel 计算后 Concat。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KIVI 的 tiled mixed-precision attention：

```
// Attention Score = Concat(tiled sub-results)
A_g = t_Q @ Dequant(Q(X_K_g))^T   // Tile 1: Q_MatMul kernel (fused dequant+matmul)
A_r = t_Q @ X_K_r^T               // Tile 2: Standard FP16 matmul
A = Concat([A_g, A_r], dim=token) // Concatenate along token dimension

// Attention Output = sum of tiled sub-results
Softmax_split:
    A_g_sm = Softmax(A)[:, :-R]   // normalized weights for grouped part
    A_r_sm = Softmax(A)[:, -R:]   // normalized weights for residual part

t_O_g = A_g_sm @ Dequant(Q(X_V_g))   // Tile 1: Q_MatMul for grouped value
t_O_r = A_r_sm @ X_V_r               // Tile 2: Standard matmul for residual value
t_O = t_O_g + t_O_r                  // Sum (not concat)
```

关键设计点：
- **Token 维度拆分**：both key 和 value 沿 token 维度拆分为 grouped + residual，tiled matmul 在 token 维度分块
- **Softmax 跨 tile 归一化**：softmax 必须跨全部 token 执行，因此先在拼接后的 A 上 softmax，再按 grouped/residual 分割
- **Output 为 Sum 而非 Concat**：attention output 在 hidden dim 上未拆分，两个 tile 的结果直接相加

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 在 CUDA 层面实现了两种 kernel 的无缝调度：(1) Q_MatMul 处理 grouped 量化 tile；(2) cuBLAS GEMM 处理 residual FP16 tile。tile 拆分和拼接在 PyTorch 层面通过 tensor slicing 完成。类似的分块策略也被 FlashAttention 用于处理长序列（分块 softmax），以及 vLLM 的 PagedAttention（分页处理 KV cache block）。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---

## Chunkwise Recurrent Representation (for Gated Retention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunkwise Recurrent Representation（分块循环表示）是 Gated Retention 的第三种计算范式，统一了 Parallel 和 Recurrent 两种表示。它将序列划分为大小为 B 的 chunk（如 B=256），每个 chunk 内使用 Parallel 计算（利用 Tensor Core 的矩阵乘加速），chunk 间使用 Recurrent 计算（通过 state R_i 传递跨 chunk 的历史信息）。数学上输出分为 Inner-Chunk 部分（(Q_{[i]}K_{[i]}^T⊙D_{[i]})V_{[i]}，chunk 内标准并行计算）和 Cross-Chunk 部分（(Q_{[i]}R_{i-1})⊙β_{[i]}，利用上一 chunk 的 state）。此范式在 FLOPs 上优于 fully parallel（避免计算上三角全部元素）且在迭代数上优于 fully recurrent（B 倍减少）。在 YOCO 中，prefill 阶段使用 chunkwise（固定 chunk_size=256），decode 阶段切换到纯 recurrent。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 YOCO Triton kernel 中 chunkwise gated retention 的计算为例：

```python
def chunkwise_gated_retention(Q, K, V, gt, past_kv, chunk_size=256):
    """
    Q, K, V: [batch, heads, seq_len, dim]
    gt: [batch, heads, seq_len]  -- data-dependent gate γ
    past_kv: [batch, heads, dim, dim]  -- previous chunk's state R
    """
    B, H, N, D = Q.shape
    num_chunks = N // chunk_size
    
    # Compute decay in log-space for numerical stability
    log_gamma = F.logsigmoid(gt) / gate_logit_normalizer  # [B,H,N]
    
    output_chunks = []
    for i in range(num_chunks):
        start, end = i*chunk_size, (i+1)*chunk_size
        Q_c, K_c, V_c = Q[:,:,start:end], K[:,:,start:end], V[:,:,start:end]
        log_g_c = log_gamma[:,:,start:end]  # [B,H,chunk]
        
        # ---- Cross-Chunk (recurrent part) ----
        # cumulative decay over this chunk: β = exp(cumsum(log_gamma))
        cumdecay = log_g_c.cumsum(dim=-1)  # [B,H,chunk]
        beta = cumdecay.exp()               # multiplicative decay
        
        # Cross-chunk output: (Q @ past_kv) * exp(decay_from_start)
        cross_out = (Q_c @ past_kv) * beta.unsqueeze(-1)  # [B,H,chunk,D]
        
        # ---- Inner-Chunk (parallel part) ----
        # Causal decay mask within chunk
        decay_mask = (cumdecay.unsqueeze(-1) - cumdecay.unsqueeze(-2)).exp()
        causal_mask = torch.triu(torch.ones(chunk, chunk), diagonal=1) * -1e9
        D_c = decay_mask + causal_mask.to(Q.device)
        
        attn_scores = (Q_c @ K_c.transpose(-1,-2)) * D_c   # [B,H,chunk,chunk]
        inner_out = attn_scores @ V_c                       # [B,H,chunk,D]
        
        # ---- Combine & Output ----
        output_c = inner_out + cross_out
        output_c = group_norm(output_c)
        output_chunks.append(output_c)
        
        # ---- Update State for Next Chunk ----
        chunk_decay = beta[:,:,-1].unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
        value_decay = (beta[:,:,-1].unsqueeze(-1) - cumdecay).exp()  # [B,H,chunk]
        current_kv = chunk_decay * past_kv + K_c.transpose(-1,-2) @ (V_c * value_decay.unsqueeze(-1))
        past_kv = current_kv
    
    return torch.cat(output_chunks, dim=2), past_kv
```

**Annotations**: chunk_size=256 是平衡点——太小则 recurrent 迭代次数多、太大则 parallel 部分 O(B²) 开销增大。gate_logit_normalizer 从训练数据统计得出，用于将 sigmoid 输出映射到合适的 decay 范围。past_kv ∈ R^{d×d} 是唯一跨 chunk 传输的状态矩阵（O(d²) 内存）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Chunkwise Recurrent 在 YOCO 中通过 Triton kernel 实现（基于 FLA 库：https://github.com/sustcsonglin/flash-linear-attention）。具体使用：(1) **Prefill 阶段**——输入长序列（可能 512K tokens），用 chunkwise 减少内存和 FLOPs；(2) **训练阶段**——长序列训练时用 chunkwise 代替 parallel 以降低峰值显存（避免存储完整 N×N attention matrix）；(3) **与 FlashAttention 类比**——chunkwise 的效果类似于分块 attention，但通过 recurrent state 保证 chunk 间信息无损传递。限制：chunkwise recurrent 在 chunk 边界处的精度依赖 gate_logit_normalizer 的校准；跨 chunk 的 state R 在长序列中可能积累数值误差。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

## Chunk Parallelism (for Long-Sequence Training of YOCO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunk Parallelism 是 YOCO 为分布式长序列训练提出的并行策略，利用 Cross-Decoder 解耦注意力依赖的特性来减少 GPU 通信开销。在标准序列并行中，序列被分割到多个设备，每层 self-attention 都需要 all-gather 通信来交换 KV。YOCO 的 Chunk Parallelism 将序列切分为多个 chunks 分配到不同 GPU：Self-Decoder 仅需在相邻设备间传递边界信息（如 gated retention 的 recurrent state S 或 sliding-window 的边界 tokens）；Cross-Decoder 的 K̂,V̂ 则仅需**一次** all-gather（而非每层一次），因为所有 cross-decoder 层共享同一组缓存。这大幅降低了通信频率、减少了 GPU memory fragmentation，使 YOCO 在极长序列训练时具可扩展性优势。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Chunk Parallelism 在两 GPU 上的训练流程：

```
Sequence [x_1, ..., x_{2N}] → Split into two chunks

GPU 0: [x_1, ..., x_N]        GPU 1: [x_{N+1}, ..., x_{2N}]

=== Self-Decoder (per-device, with boundary communication) ===
for layer in self_decoder_layers:     # layers 1..L/2
    # GPU 0 sends last tokens of its chunk to GPU 1 (for window/recurrent state)
    # GPU 1 receives boundary state from GPU 0
    # Each GPU computes efficiently within its chunk
    X_0 = SelfDecoderLayer(X_0)       # local chunk
    X_1 = SelfDecoderLayer(X_1)       # local chunk
    # Communication volume: O(C*d) for sliding-window or O(d²) for retention

# Output: M_0 ∈ R^{N×d}, M_1 ∈ R^{N×d}

=== Generate Global KV Cache (one-time all-gather) ===
K̂_0 = proj_K(M_0), V̂_0 = proj_V(M_0)  # local computation
K̂_1 = proj_K(M_1), V̂_1 = proj_V(M_1)

# All-gather K̂, V̂ across all devices — ONLY ONCE!
K̂ = AllGather([K̂_0, K̂_1])  # concatenated [2N, d] on both GPUs
V̂ = AllGather([V̂_0, V̂_1])

=== Cross-Decoder (K̂,V̂ already replicated on all devices) ===
# No further communication needed for attention!
for layer in cross_decoder_layers:    # layers L/2+1..L
    Q = proj_Q(X)
    O = CrossAttention(Q, K̂, V̂)       # local, K̂,V̂ already complete
    X = SwiGLU(O)
    # Collect output only at classification head
```

**Annotations**: Self-Decoder 的通信量受高效 attention 限制（sliding-window: O(C×d), gated retention: O(d²)），远小于全局 attention 的 O(N×d)。Cross-Decoder 的 all-gather 仅传输 K̂,V̂（O(N×d)），仅一次。对比标准 Transformer 每层都需要 all-gather Q,K,V（O(L×N×d)），chunk parallelism 减少了约 L× 的通信量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Chunk Parallelism 的实现和使用：(1) 基于 SuperScaler 或 Megatron-LM 的序列并行框架实现；(2) 适用于极长序列训练场景（百万 token 级别）；(3) Self-Decoder 的边界通信可以使用 P2P send/recv（比 all-gather 更高效）；(4) Chunk 数量可以动态调整——chunk 越多则每设备序列越短（内存节省），但边界通信总量增加。论文未开源 Chunk Parallelism 的实现代码，仅描述了算法原理。限制：需要 YOCO 架构（Cross-Decoder 共享 KV cache）；Self-Decoder 边界通信对于 sliding-window 很简单（直接传 tokens），对于 gated retention 需要传 state S（O(d²) which 在 head_dim 较大时可能成为瓶颈）。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)

## On-the-fly Weight Dequantization (Software/GPU)

## On-the-fly Weight Dequantization (Software/GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
On-the-fly Weight Dequantization（在线权重反量化）是 TinyChat/AWQ 推理系统中将 4-bit 量化权重在 GPU/CUDA kernel 内部实时恢复为 FP16 的技术。与传统的"先反量化到显存再计算"不同，on-the-fly 方式在 GEMM/GEMV 主循环内部完成反量化——权重从 DRAM 以 packed INT4 格式读取到寄存器，在寄存器中解包并乘以 group-wise Δ（量化 scale），得到 FP16 权重值后立即参与 FMA 运算，然后丢弃（不写回 DRAM）。这样避免了将 4× 数据量的 FP16 反量化权重写回 DRAM，将 decode 阶段的 arithmetic intensity 从 ≈1 提升至 ≈4 FLOPs/Byte（RTX 4090 上峰值性能上限从 ~1 TFLOPS 升至 ~4 TFLOPS）。TinyChat 同时为矩阵-矩阵乘（prefill）和矩阵-向量乘（decode）实现了融合 dequantization kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 Llama-2-7B decode 阶段（batch_size=1）的 W4A16 on-the-fly dequantization GEMV kernel 为例：
```cuda
// TinyChat CUDA kernel: fused INT4-dequant + FP16-GEMV
// 输入: packed_weights (INT4, 每 2 weights/byte), scales (FP16, per-group)
// 输入: activation [1, C_in] FP16
// 输出: output [1, C_out] FP16

__global__ void gemv_w4a16_dequant_fused(
    const uint8_t* packed_w,   // [C_out, C_in/2]
    const half* scales,        // [C_out, C_in/group_size]
    const half* input,         // [C_in]
    half* output,              // [C_out]
    int C_in, int C_out, int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // output row
    if (row >= C_out) return;

    float acc = 0.0f;
    int group_id = 0;

    for (int j = 0; j < C_in; j += 2) {
        // Step 1: 加载 packed byte
        uint8_t byte = packed_w[row * (C_in/2) + j/2];
        
        // Step 2: 解包两个 4-bit 权重
        int8_t w0 = (int8_t)(byte & 0x0F);       // 低 4-bit
        int8_t w1 = (int8_t)((byte >> 4) & 0x0F); // 高 4-bit
        // INT4 有符号范围: [-8, 7], 解包时做 sign extension
        
        // Step 3: 加载 group scale 并反量化
        if (j % group_size == 0) {
            half scale = scales[row * (C_in/group_size) + group_id++];
        }
        half w0_fp = __half2float(w0) * scale;
        half w1_fp = __half2float(w1) * scale;
        
        // Step 4: FMA 累加 (读到即算，不存回 DRAM)
        acc += w0_fp * __half2float(input[j]);
        acc += w1_fp * __half2float(input[j+1]);
    }
    output[row] = __float2half(acc);
}
```

关键设计决策：
- 权重读入寄存器后立即解包 → 反量化 → FMA，中间结果仅存于寄存器，不写回 shared memory / DRAM
- Group scale 按需加载（每 group_size=128 个权重加载一次），不占用过多寄存器
- 对于 batch_size > 1 的 GEMM 场景（prefill），可复用解包后的权重到 shared memory 中供多个 activation row 使用

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TinyChat 在 PyTorch 中通过 CUDA extension 实现 fused dequantization kernel。实现方式：(1) 使用 PyTorch 的 `torch.utils.cpp_extension.load_inline` 或 setuptools 编译 CUDA kernel；(2) 在模型 forward pass 中用 autograd Function 包装自定义 kernel；(3) 支持 FP16 和 BF16 两种 activation 精度。关键工程实践：针对不同 GPU 架构使用不同的 warp tile 大小——RTX 4090 (SM89) 使用 128-thread per output row，Jetson Orin (SM87) 使用 64-thread。TinyChat 的 on-the-fly dequantization 在 4090 上实现 ~194 tokens/s (Llama-2-7B)，相比 HuggingFace FP16 的 52 tokens/s 加速 3.7×。代码：https://github.com/mit-han-lab/llm-awq/tree/main/tinychat。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

---

## SIMD-aware Weight Packing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SIMD-aware Weight Packing 是 TinyChat 针对 CPU SIMD 架构（ARM NEON、x86 AVX）提出的 INT4 权重排列优化策略。传统顺序排列（w0, w1, ..., w31）下，每个 4-bit 权重的解包需要 3 条标量指令（shift + AND + FMA scaling），32 个权重共 96 条标量指令。SIMD-aware packing 将权重重新排列为交错顺序，使得单条 SIMD 指令可并行解包整个寄存器宽度的权重。以 ARM NEON 128-bit 为例：将 32 个 4-bit 权重排列为 (w0, w16, w1, w17, ..., w15, w31)，一个 128-bit 寄存器可同时解包全部 32 个权重，仅需 3 条 SIMD 指令（AND 提取低位、shift+AND 提取高位、FMA scaling）。通用规则：对于 2^n-bit SIMD 寄存器，相邻权重的索引差为 `1/8 × 2^n`，因为每个寄存器可存 `1/8 × 2^n` 个 8-bit 解包后的权重。GPU 端采用不同排布：每 8 个权重打包为 (w0, w2, w4, w6, w1, w3, w5, w7) 顺序（参照 Kim et al., 2022）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ARM NEON 128-bit SIMD-aware unpacking 的伪代码：
```c
// 传统顺序 packing: [w0|w1|w2|w3|...|w31] (每 4-bit 一个 weight)
// 每个权重需要独立的标量指令解包，无 SIMD 利用

// SIMD-aware packing: [w0|w16|w1|w17|w2|w18|...|w15|w31]
// 128-bit 寄存器 = 32 个 4-bit weight indices

// 解包过程（3 条 NEON SIMD 指令）：
uint8x16_t packed = vld1q_u8(packed_weights_ptr);  // 加载 128-bit

// Step 1: 提取低 4-bit (w0, w16 的低 4bit, w1, w17 的低 4bit, ...)
uint8x16_t low_nibbles = vandq_u8(packed, vdupq_n_u8(0x0F));

// Step 2: 提取高 4-bit (w0, w16 的高 4bit, w1, w17 的高 4bit, ...)
uint8x16_t high_nibbles = vshrq_n_u8(packed, 4);

// Step 3: 查表 LUT + FMA scaling
// 将解包后的 indices 用作 LUT 索引，查得 FP16 值后乘以 scale
// 使用 NEON FMA 指令一次性完成乘加

// 对比标量方法: 32 weights × 3 instructions = 96 instructions
// SIMD 方法: 3 SIMD instructions, 理论加速 32×
```
实际收益：ARM CPU 上 SIMD-aware packing 额外提供 ~1.2× 加速（相比未做 SIMD-aware packing 的 bulk dequantization）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：(1) 量化后的 INT4 权重在保存时按目标平台的 SIMD 宽度进行重排（offline 预处理）；(2) 不同平台使用不同的 packing layout——ARM NEON 128-bit 用 32-weight 交错，AVX 256-bit 用 64-weight 交错，GPU 用 8-weight 交错；(3) 推理时直接按 packed layout 加载，用对应 SIMD intrinsic 解包。TinyChat 在 CPU 后端（C++ 实现）中针对 ARM NEON 和 x86 AVX 分别实现了 SIMD-aware packing 和解包 kernel。代码：https://github.com/mit-han-lab/llm-awq。这种技术的通用性：任何需要在 CPU SIMD 上执行低比特推理的 weight-only 量化方案（GGUF/Q4_0, GPTQ, AWQ 等）都可受益。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

在 Squat 中，INT4 Concatenation 是 SIMD-aware packing 的扩展——将相邻行4-bit权重拼接入16-bit寄存器（不零扩展到8-bit），利用 ARM `mla` 指令（16-bit×16-bit→32-bit累加器）在单指令完成两个子字节乘加。与 AWQ 的交错排列不同，Squat 的拼接策略专注于最大化 SIMD 寄存器的位宽利用率（100% vs 零扩展方案的50%），理论计算量减半。配合 bit-shift + row-wise summation 恢复正确结果。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT Dequantization（查找表反量化）是一种将量化权重从低比特格式恢复到高精度格式的 GPU kernel 实现技术。当量化格式的值不是均匀分布（如 NormalFloat、FP 格式），无法通过简单的代数公式（如 `w_fp16 = scale * (w_int - zero_point)`）完成反量化时，需要使用预先存储的查找表（LUT）将每个量化索引映射到对应的浮点值。AFPQ 论文在推理系统中使用 LUT 来完成 NF4/NF3 值到 FP16 的转换：NF 格式的 16 个（NF4）或 8 个（NF3）候选值预先存储在 GPU 的 constant memory 或 register 中，反量化时通过量化索引查表得到对应的 FP16 值。LUT 之后，再用 scale_pos/scale_neg 进行非对称缩放得到最终 FP16 权重。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AFPQ 论文中 NF4-asym LUT dequantization kernel 的执行过程：
```
# GPU Kernel: NF4-asym Dequantization
# 输入: packed_4bit_weights (byte array), scales_pos (FP16), scales_neg (FP16)
# 常量: NF4_LUT[16] = [-1, -0.6962, ..., 0.7230, 1.0]  # 16 个 FP16 值

# Constant Memory 中预存 LUT
__constant__ half NF4_LUT[16] = {...};

__global__ void dequant_nf4_asym_kernel(
    uint8_t* packed_w,    // 每 2 个 NF4 → 1 byte
    half* scales_pos,     // FP16, 每组一个
    half* scales_neg,     // FP16, 每组一个
    half* w_fp16,         // 输出: 反量化后的 FP16 权重
    int group_size,       // = 128
    int num_groups
) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    
    // 加载当前 group 的 scale
    half s_pos = scales_pos[group_id];
    half s_neg = scales_neg[group_id];
    
    // 每个线程处理多个元素
    for (int i = tid; i < group_size; i += blockDim.x) {
        int byte_idx = group_id * (group_size / 2) + i / 2;
        uint8_t byte = packed_w[byte_idx];
        
        // 提取两个 NF4 索引
        uint8_t idx;
        if (i % 2 == 0) idx = byte & 0x0F;
        else            idx = (byte >> 4) & 0x0F;
        
        // LUT 查找
        half val = NF4_LUT[idx];
        
        // 非对称反量化
        if (val > 0)       w_fp16[...] = s_pos * val;
        else if (val < 0)  w_fp16[...] = s_neg * val;
        else               w_fp16[...] = 0;
    }
}
```
关键设计：(1) LUT 存在 constant memory 中（所有线程同时读取同一地址时零延迟）；(2) packed 4-bit 格式每 byte 存 2 个权重，减少显存带宽；(3) 非对称 scale 的 branch 基于 val 的符号（非判断 weight 原始符号），避免额外存储符号 bit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LUT dequantization 的实现方式：(1) CUDA constant memory——适合 LUT 大小 ≤ 64KB（NF4 LUT 仅 32 bytes）；(2) Shared memory——当 LUT 较大或不同 warp 需不同 LUT 时使用；(3) Register——极小的 LUT（如 NF3 的 8 个值）可展开到寄存器。在 AFPQ 的 FasterTransformer 集成中，NF4-asym kernel 替换了原有的 INT4 dequant kernel（后者通过 `w_fp16 = scale * (w_int - zero_point)` 的代数计算完成，无需 LUT）。LUT 方法的局限性：(1) 适用于非均匀量化格式（NF、FP），但比 INT 的代数反量化多一次 memory read；(2) 可能增加 register pressure（如果 LUT 被编译器展开到寄存器）。AFPQ 论文观察到 NF4-asym 推理延迟（265ms）高于 INT4（174ms），部分由 LUT 和额外的 scale branch 导致，并指出可以通过 kernel 优化缩小差距。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs

---

## LUT-GEMM (Lookup Table based GEMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-GEMM 是一种利用查找表（Lookup Table）加速二值/低比特量化权重 GEMM 的 GPU kernel 优化方法。核心思想：当权重矩阵是二值（{-1,+1}）时，与激活向量的内积退化为对激活元素的加/减操作。将权重按每 μ 个比特分组（通常 μ=3~4），预计算这 2^μ 种二值模式与激活片段的所有可能内积结果存入 LUT。实际计算时，用 μ-bit 权重模式作为索引直接查表获取结果，避免了逐个浮点乘加。NAVER 的 LUT-GEMM (Park et al., 2022) 首次将 BCQ 格式下的 LUT 计算实现在 GPU 上，支持 uniform 和 non-uniform (BCQ) 两种量化方案。后续工作：FLUTE (MIT/CMU, EMNLP 2024) 通过 LUT 向量化和跨 shared memory bank 复制消除 bank conflicts，实现 2-4× GEMM 加速；LUT Tensor Core (Microsoft, ISCA 2025) 提出 dedicated LUT-based Tensor Core 硬件设计；FIGLUT (POSTECH+NAVER, HPCA 2025) 设计 custom RAC（Read-Accumulate）单元替代 MAC。在 AnyBCQ 中，LUT-GEMM 思想被用于自研 CUDA kernel：每个比特平面 B_i ∈ {-1,+1} 的 GEMM 通过 LUT 加速加减操作，p 个比特平面结果乘以 α_i 后累加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LUT-GEMM 的基本伪代码（以 μ=4 为例）：

```
# 输入: activation A ∈ R^{1×K}, binary weight B_i ∈ {-1,+1}^{K×N}, μ=4
# 每 μ 个连续 K 维度为一组，构建 2^μ=16 entry LUT

for n in range(0, N, TILE_N):  # 输出 tile
    for k_tile in range(0, K, TILE_K):  # 输入 tile
        # Step 1: 构建 LUT (per tile)
        A_tile = A[k_tile : k_tile+TILE_K]  # 激活片段
        LUT = [0] * (1 << μ)  # 2^μ entries
        for g in range(0, TILE_K, μ):  # 每 μ 个元素一组
            for pattern in range(1 << μ):
                # pattern 的 μ bits 决定加减模式
                sum_val = 0
                for bit in range(μ):
                    sign = 1 if (pattern >> bit) & 1 else -1
                    sum_val += sign * A_tile[g + bit]
                LUT[pattern] += sum_val
        
        # Step 2: 查表获取 partial sums
        for n_tile in range(0, N, TILE_N):
            for g in range(0, TILE_K, μ):
                w_bits = B_i[k_tile+g : k_tile+g+μ, n_tile]  # μ-bit pattern
                output[n_tile] += LUT[w_bits]  # 直接查表！
```

GPU 上的关键瓶颈：多个线程同时访问 LUT 的不同 entry 时产生 shared memory bank conflicts。解决方案：(1) FLUTE 的 LUT 跨 bank 复制（每 bank 存完整 LUT）；(2) T-MAC 的 in-register table（使用 ARM TBL / x86 PSHUF 指令避免 shared memory）；(3) FIGLUT 的 custom decoding 硬件消除 bank conflicts。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LUT-GEMM 的开源实现：(1) github.com/naver-aics/lut-gemm：NAVER 的官方实现，支持 BCQ + uniform 量化；(2) FLUTE (MIT)：CUDA kernel library，2-4× GEMM 加速；(3) T-MAC (Microsoft)：CPU 端实现，in-register LUT。在 AnyBCQ kernel 中，LUT-GEMM 用于每个比特平面的 GEMM 加速，p 个平面的 LUT-GEMM 结果按 α_i 加权累加。GPU 实现要点：LUT 大小 = 2^μ × sizeof(FP16)，μ=4 时 32 bytes，适合 constant memory 或寄存器；μ 越大查表次数越少但 LUT 越大（μ=5: 64 bytes, μ=6: 128 bytes）。权衡：LUT-GEMM 在 memory-bound 场景（decode, batch=1）收益最大，因减少的算术操作等价于更低的 arithmetic intensity 要求。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---

## Bit-Transpose (in Quantization Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-Transpose（比特转置）是 Any-Precision LLM 等非均匀多精度量化 kernel 中的一种数据重排操作。在非均匀量化中，每个权重存储为 centroid index（如 4-bit 模型的 0-15），推理时以 p 个独立比特平面格式存储和加载（每平面 1 bit/元素）。为了将 p 个独立的 1-bit 平面重组为一个 p-bit 的 centroid index 用于 table lookup，需要将 p 个平面"转置"——即对每个权重位置，从 p 个比特平面中分别取出对应 bit，合并为 p-bit 整数索引。在 GPU 上，这个操作涉及大量的 bitwise shift + OR 运算和跨比特平面的不规则内存访问。AnyBCQ 论文（Table 7）的 kernel 延迟分解显示：bit-transpose 是 Any-Precision LLM kernel 的最大开销来源，占 kernel 总延迟的 35-58%（取决于矩阵形状和比特宽度），远高于 centroid table lookup 的 9-17%。AnyBCQ 通过使用 BCQ 二值格式彻底消除了 bit-transpose——BCQ 的比特平面直接是可计算操作数（{-1,+1}），无需转置为 index。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Any-Precision LLM kernel 中 bit-transpose 的计算过程：

```
# 输入: p 个比特平面 BP_0, BP_1, ..., BP_{p-1}
# 每个 BP_i 是 M×K 的 packed binary tensor (1 bit/元素)
# 输出: index_matrix ∈ {0,...,2^p-1}^{M×K}

for m in range(M):
    for k in range(K):
        index = 0
        for i in range(p):  # p=2/3/4
            bit = extract_bit(BP_i, m, k)  # 从第 i 个平面取第 (m,k) 位置的 bit
            index |= (bit << i)             # 移位合并
        index_matrix[m, k] = index  # 0~2^p-1 的 centroid index

# 然后用 index_matrix 查 centroid table:
# weight_deq[m,k] = centroid_table[index_matrix[m,k]]
```

GPU 实现中的瓶颈：(1) 每个权重的 p 个 bit 来自 p 个不同比特平面，访问内存位置不同（跨平面、非连续）；(2) 提取并合并 p 个 bit 需要 p 次 loaded bit + p-1 次 shift + p-1 次 OR（或等效的 bitwise 操作）；(3) 输出 index 后续还要用于 shared memory / global memory 中的 centroid table lookup，形成两阶段依赖。Table 7 数据显示，bit-transpose 占比随 K 增大而增加（K=14336 时最高 57.71%），说明不规则内存访问是主要瓶颈而非纯计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit-transpose 的优化方案：(1) SIMD 加速——使用 CUDA 的 `__byte_perm()` 或 warp shuffle 指令批量执行 bit 重组；(2) 预转置存储——在模型加载时预先将比特平面存为"按权重交织"的格式（每个 byte 存某权重的 p bits），消除推理时的 transpose 开销，但代价是存储格式与"按需只加载 p 个平面"的带宽节约目标冲突；(3) 消除法——AnyBCQ 的方法：不使用 centroid index 格式，直接用 BCQ 二值平面操作，从根源消除 transpose 需求。BCQ 的比特平面本身就是操作对象（±1 乘激活），不经过 "index → centroid" 的中间表示。这也是 BCQ 在多精度场景下硬件效率优于非均匀量化的根本原因。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---

## XNOR and Bit-count Operations（同或与位计数运算，1-bit 卷积硬件实现）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XNOR 和 bit-count（popcount）是二值化神经网络（BNN）中将浮点卷积乘加运算（MAC）替换为纯位操作的核心计算原语。当权重和激活都被二值化为 {+1, -1}（通常映射为 {1, 0}）后，卷积中的逐元素乘法退化为同或（XNOR）运算：`w=+1, a=+1 → +1·+1=+1, XNOR(1,1)=1`；`w=+1, a=-1 → +1·-1=-1, XNOR(1,0)=0`。即 XNOR 输出 1 时等价于乘法结果 +1，输出 0 时等价于 -1。累加操作退化为位计数（popcount）：统计 XNOR 结果中 1 的个数（得 +1 计数），最终求和 = `2*popcount(XNOR(x_b, w_b)) - n`。硬件上，XNOR 为单比特门操作（vs FP16 MAC 需多周期浮点单元），popcount 可用专用指令（x86 POPCNT, CUDA __popc）或 LUT 高效实现。理论加速比：32x 内存节省（32-bit → 1-bit），64x 能量节省（浮点乘加 → 位操作）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 BI-DiffSR 的二值化卷积 kernel 概念为例（论文未给出定制 CUDA kernel，以下为通用 1-bit Conv 设计）：
```cuda
// 1-bit 卷积 kernel 概念: bit-packed GEMM
// packed 格式: 每 byte 存 8 个 1-bit 值 (1=+1, 0=-1)
__global__ void binarized_conv2d(
    const uint8_t* packed_act,    // 1-bit 激活, packed
    const uint8_t* packed_w,      // 1-bit 权重, packed
    int32_t* output,              // int32 累加结果
    int H, int W, int C_in_pack, int K,
    float w_scale                  // 权重缩放因子 ||w||_1/n
) {
    int h_idx = blockIdx.y, w_idx = blockIdx.x;
    int c_out = threadIdx.x;
    int accum = 0;
    
    for (int cp = 0; cp < C_in_pack; cp++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                uint8_t act_byte = packed_act[h_idx][w_idx][cp];
                uint8_t w_byte = packed_w[c_out][cp][kh][kw];
                
                // XNOR: 按位同或, ~(a ^ b)
                uint8_t xnor_result = ~(act_byte ^ w_byte);
                
                // popcount: CUDA intrinsic __popc (32-bit)
                int ones = __popc((uint32_t)xnor_result);
                accum += 2 * ones - 8;  // ones - (8-ones)
            }
        }
    }
    output[h_idx*W + w_idx][c_out] = accum;
    // output * w_scale (后续 FP 乘法恢复量级)
}
```
关键优化：(1) 每 byte 操作同时处理 8 个 1-bit 值；(2) XNOR 和 popcount 均在寄存器完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPU（x86）：`_mm512_xor_si128` + `_mm_popcnt_u64` / `std::popcount`（C++20）。CUDA：`__vpopcnt4()`（SM 8.0+, 32-bit）、`__popcll()`。已知开源框架：daBNN（ARM NEON）、Larq Compute Engine（TF Lite）、BNN-PYNQ（FPGA）。PyTorch 无原生 1-bit kernel，BI-DiffSR 未实现定制 CUDA kernel（仅标注理论加速比），实际部署需自定义 CUDA kernel 或导出到 BNN 专用推理引擎。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---

## QuTLASS

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QuTLASS 是 MR-GPTQ 论文配套发布的高性能低精度量化 kernel 库，基于 NVIDIA CUTLASS 构建，专为 Blackwell GPU（SM100/SM120）优化。提供两类 kernel：(1) Quantization-related kernels——轻量级 fused kernel 实现在线 block-wise 旋转 + 量化 + scale 计算的融合，支持 k∈{16,32,64,128} 的 block diagonal 矩阵旋转，通过自定义 epilogue function 将量化/scale 直接集成进变换 kernel；(2) Matmul-related narrow precision kernels——处理 FP4 量化与 tcgen05.mma 矩阵乘间的 scale 重排（硬件强制的 block scaling factors layout），通过 Triton kernel 实现，支持 CUTLASS 和 FlashInfer 多后端灵活切换。

QuTLASS 的关键设计：对 k<256 的 block 旋转，dense 变换为 memory-bound，因此旋转矩阵可以运行时从内存加载（任意矩阵，不限于 Hadamard），所有旋转矩阵几乎同成本。量化方法通过模板设计支持 MSE 和 Abs-Max，便于扩展。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuTLASS 中 FP4 推理的完整 kernel 执行流程：
```
// Kernel 1: Fused Online Rotation + Quantization (CUDA)
// 输入: FP16 activation X [M, K], block-diag matrix H_k in memory
// 输出: MXFP4 quantized activation X_q + per-group scales
__global__ void fused_rotate_quantize_mxfp4(
    half* X, half* H_k,        // H_k: k×k per block, row-major
    uint8_t* X_q_packed,        // 输出: E2M1 packed (2×4-bit/byte)
    uint8_t* scales_E8M0,       // 输出: E8M0 power-of-two scales
    int M, int K, int k         // k = group_size = 32
) {
    // Step 1: Block-wise Hadamard rotation (memory-bound for k<256)
    // X_rot[b, :k] = X[b, :k] @ H_k  per block
    // k 小时每个 thread 处理的 FLOPs 少于 bytes loaded
    
    // Step 2: Per-group absmax scale (fused epilogue)
    // s_G = max(|X_rot[b, g*k : (g+1)*k]|) for each group g
    // s_G_q = round_to_power_of_two(s_G)  // E8M0 quantization
    
    // Step 3: E2M1 quantization (fused epilogue)
    // x_norm = X_rot / s_G_q
    // x_fp4 = RTN_E2M1(x_norm)   // 4-bit E2M1 format
    // Pack 2×4-bit into 1 byte
}

// Kernel 2: Scale Rearrangement (Triton)
// 输入: per-group scales in natural order
// 输出: scales rearranged for tcgen05.mma layout
// 原因: NVIDIA Blackwell tcgen05.mma 要求特定的 block scaling factors layout
// 参照 cuBLAS doc: block-scaling-factors-layout

// Kernel 3: FP4 Matrix Multiplication (CUTLASS/FlashInfer backend)
// 输入: W_q (MXFP4 packed), X_q (MXFP4 packed), rearranged scales
// 调用: tcgen05.mma (Blackwell hardware instruction)
// 输出: FP16/BF16 output activation
```

B200 单层 throughput 结果（Llama-3.3-70B 典型层形状）：
- "Ideal": 仅 FP4 matmul（不含旋转/量化/scale 开销）= ~4× vs FP16
- "Actual" (QuTLASS): 含全部开销 = ~3.6× vs FP16（MXFP4）
- MXFP4 比 NVFP4 高 ~15% throughput（power-of-two scales 降低硬件开销）
- RTX 5090: 6× layer-wise（ideal 8×），4× end-to-end

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/IST-DASLab/qutlass。基于 CUTLASS（https://github.com/NVIDIA/cutlass）构建，利用 CUTLASS 的 epilogue fusion 机制将量化和 scale 操作融合进 rotation kernel。FlashInfer 后端支持（https://github.com/flashinfer-ai/flashinfer）。Triton kernel 用于 scale rearrangement（硬件强制的 layout 转换）。支持 MXFP4 和 NVFP4 两种格式。集成进 vLLM 进行端到端推理评估：Llama-3.3-70B B200 端到端 2.2× speedup vs BF16，RTX 5090 端到端 4× speedup。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization

---

## Fused Online Rotations (for FP4 Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Online Rotations 是 QuTLASS 中的一种 GPU kernel 优化技术，将激活端的 block-wise 旋转（如 Hadamard）与后续的 FP4 量化操作融合为单个 kernel，消除中间 DRAM 写入。核心原理：对 k < 256 的 block diagonal 变换，dense 矩阵乘法为 memory-bound（arithmetic intensity 极低），因此旋转本身跟 memory copy 几乎同开销。通过将量化（absmax/MSE scale calculation + E2M1 RTN）作为 custom epilogue function 直接融合进旋转 kernel（利用 CUTLASS epilogue fusion），旋转输出不写回 DRAM，直接生成 FP4 packed 格式和 scales。这使得 MR-GPTQ 的 "micro-rotation" 组件（激活端在线 Hadamard 旋转）几乎零额外开销——与标准 FP4 量化（无旋转）的延迟差异在测量噪声范围内。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 传统方法（未融合，3 次 kernel launch + 2 次 DRAM 写入）:
// Kernel 1: X_rot = X @ H_k  → write X_rot to DRAM
// Kernel 2: s_G = absmax(X_rot per group) → write s_G to DRAM
// Kernel 3: X_q = E2M1_quantize(X_rot / s_G) → write X_q to DRAM

// Fused 方法（QuTLASS，1 次 kernel launch + 0 次 DRAM 写入）:
__global__ void fused_rotate_quantize(
    half* X, half* H_k,         // inputs from DRAM
    uint8_t* X_q_out,           // output: FP4 packed (direct to DRAM)
    uint8_t* scales_out         // output: E8M0/E4M3 scales (direct to DRAM)
) {
    // All intermediate results stay in registers / shared memory
    
    // Phase 1: Block rotation (memory-bound, k<256)
    // Thread block loads X tile and H_k tile from DRAM
    // Computes X_rot = X_tile @ H_k_tile in shared memory
    
    // Phase 2: Fused epilogue — quantization (no intermediate DRAM write)
    // Directly from X_rot in shared memory:
    //   s_G = absmax(X_rot per group of k elements)
    //   s_G_q = quantize_scale(s_G)  // E8M0 or E4M3
    //   x_norm = X_rot / s_G_q
    //   x_q = RTN_E2M1(x_norm)      // FP4 E2M1
    //   pack 2×4-bit into 1 byte
    
    // Write packed X_q and scales to DRAM (single write each)
}

// 关键参数: k < 256 → arithmetic intensity 极低
// 旋转 cost ≈ 加载 H_k 的 memory cost (not compute-bound)
// → 任意旋转矩阵（不限于 Hadamard）几乎同成本
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 NVIDIA CUTLASS 的 epilogue fusion 机制实现：主循环执行 tile-based GEMM（X @ H_k），epilogue 直接调用量化函数处理输出 tile（不写回 global memory）。模板设计允许自定义 epilogue（MSE scale 优化、Abs-Max 量化等）。代码位于 QuTLASS（https://github.com/IST-DASLab/qutlass）。B200 实测效果："Actual"（含全部开销）与 "Ideal"（纯 matmul）的差距在 MXFP4 上仅 ~10%（3.6× vs 4× ideal），证明 fused online rotation 开销极小。

涉及论文标题：
- Bridging the Gap Between Promise and Performance for FP4 Quantization

---

## Fused Affine-Quantization Kernel（融合仿射变换-量化 Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Affine-Quantization Kernel 是 FlatQuant 基于 OpenAI Triton 实现的 GPU kernel，将 Kronecker 仿射变换 Q(P₁^T ×₁ X̃ ×₂ P₂) 融合为单个 kernel 调用。设计动机：(1) 使用 Kronecker 乘积后 P₁∈R^{n₁×n₁}、P₂∈R^{n₂×n₂} 尺寸很小（如 64×64），仿射变换为 memory-bound 操作（计算强度低）；(2) 量化也是 memory-bound。传统分开执行会产生两次全局内存往返（先写回 X' 再读取做量化）。融合后：thread block 将 P₁、P₂ 完整加载到 SRAM → slicing tile X̄∈R^{n₁×n₂} → 在 SRAM 内执行 P₁^T X̄ P₂ → 立即对结果量化 → 写回全局内存。三种 SRAM 容量场景：(a) 默认——SRAM 容纳 P₁、P₂、X̄ 及中间结果；(b) Corner Case 1——n₁ 过大，对 P₁ 非规约维 tiling；(c) Corner Case 2——n₂ 过大，分两个 kernel（先 P₁^T X̄ 写回，再乘 P₂ 并量化）。在 RTX 3090 上 hidden_dim≤14336 均使用默认设计。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 FlatQuant 默认设计（hidden_dim=4096, n₁=n₂=64, RTX 3090, batch=1）为例：

```
// Triton kernel (简化为 Python-like 伪代码)
@triton.jit
def fused_affine_quantize_kernel(
    X_ptr, P1_ptr, P2_ptr,        // FP16 inputs from DRAM
    X_q_out_ptr, scale_out_ptr,    // INT4 packed + FP16 scale → DRAM
    n1: int, n2: int, bits: int
):
    pid = tl.program_id(0)          // one program per token (k tokens)
    
    // Phase 1: Load P₁, P₂ into SRAM (once per block, shared across threads)
    P1 = tl.load(P1_ptr + offsets)  // [64, 64] FP16 → 8KB SRAM
    P2 = tl.load(P2_ptr + offsets)  // [64, 64] FP16 → 8KB SRAM
    
    // Phase 2: Load X tile into SRAM
    X_tile = tl.load(X_ptr + pid*64*64 + offsets)  // [64, 64] FP16 → 8KB SRAM
    
    // Phase 3: Affine transformation in SRAM (memory-bound)
    // X' = P₁^T @ X_tile @ P₂
    X_transformed = tl.dot(P1.T, tl.dot(X_tile, P2))  // [64, 64] in SRAM
    
    // Phase 4: Fused quantization (in SRAM, no DRAM write)
    abs_max = tl.max(tl.abs(X_transformed))
    scale = abs_max / (2**(bits-1) - 1)
    X_q_int = tl.round(X_transformed / scale)
    X_q_int = tl.clamp(X_q_int, -2**(bits-1)+1, 2**(bits-1)-1)
    X_q_packed = pack_int4(X_q_int)  // 2×4-bit → 1 byte
    
    // Phase 5: Write to DRAM (single write)
    tl.store(X_q_out_ptr + pid*32*64 + offsets, X_q_packed)  // INT4 packed
    tl.store(scale_out_ptr + pid, scale)                       // scalar FP16
```

**实测性能**（Table 6, hidden_dim=4096, batch=1, seq_len=2048 prefill / decode_1token）：
- 无融合: prefill 0.1956ms, decode 0.0184ms
- 有融合: prefill 0.0625ms, decode 0.0082ms
- 加速比: prefill 3.13×, decode 2.25×

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlatQuant 的融合 kernel 基于 OpenAI Triton 语言编写（https://github.com/openai/triton），编译为 PTX 后在 NVIDIA GPU 上运行。自定义 kernel 位于开源仓库 https://github.com/ruikangliu/FlatQuant。推理时，融合 kernel 的输出（INT4 packed 激活 + FP16 scale）直接送入 CUTLASS INT4 GEMM kernel 进行矩阵乘法。完整的预填充流程：tokens 进入 → 逐 Transformer block → (1) 层归一化 (FP16) → (2) 融合仿射量化 kernel 对激活做在线变换+量化 → (3) CUTLASS INT4 GEMM 执行量化矩阵乘法 → (4) FlashInfer kernel 对 KV cache 执行量化 → (5) 残差连接 (FP16)。端到端加速: prefill 2.30× vs FP16, decode 1.76× vs FP16。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---

## TRT-LLM Deployment for Quantized LLMs（量化 LLM 的 TRT-LLM 部署）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TRT-LLM（TensorRT-LLM）是 NVIDIA 的 LLM 推理优化框架，提供高性能 GPU kernel（特别是量化 GEMM）和完整的推理运行时。对于 INT4 量化场景，TRT-LLM 通过 CUTLASS INT4 GEMM kernel 实现对 W4A4 和 W4A8 矩阵乘法的硬件加速（利用 Tensor Core 的 INT4 MMA 指令如 mma.sync.aligned.m16n8k32）。FlatQuant 的推理栈使用 CUTLASS INT4 kernel 执行量化矩阵乘法（权重和激活均已量化为 INT4），并在 prefill 和 decoding 阶段通过 CUTLASS 的 tiling、software pipeline 和 epilogue fusion 实现高效推理。此外 FlashInfer 库用于 KV cache 量化操作。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlatQuant 中 CUTLASS INT4 GEMM 的推理流程：

```
输入:
  - A_q [M, K]: INT4 激活 (来自融合仿射量化 kernel 的输出)
  - B_q [N, K]: INT4 权重 (离线预量化)
  - scale_A [M]: per-token FP16 scale
  - scale_B [N]: per-channel FP16 scale

输出: Y [M, N] FP16

CUTLASS INT4 kernel 内部流程:
1. 从 Global Memory 加载 A_q tile, B_q tile → Shared Memory
2. 从 Shared Memory → Register (INT4 packed)
3. mma.sync.aligned.m16n8k32: 在 Tensor Core 上执行 INT4 MMA
   累积 int32 → 输出到 Register
4. Epilogue: int32 → FP16 反量化 (乘以 scale_A × scale_B)
5. Write Y tile → Global Memory
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 以 C++ 模板库形式提供（https://github.com/NVIDIA/cutlass），用户通过模板参数配置 tile size、warp tile、MMA 指令、pipeline stage 数等。FlatQuant 直接调用 CUTLASS INT4 kernel 进行量化推理。在开源实现中，通过 PyTorch 的 custom op 机制（torch.library）将 CUTLASS kernel 封装为 Python 可调用函数。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---

## Bit-shift for Power-of-Two Scaling（2 的幂次缩放中的位偏移操作）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bit-shift for Power-of-Two Scaling 是 DMQ 在 CUDA kernel 中实现的一种将 power-of-two 缩放等价转换为整数左移操作的低开销硬件优化技术。核心原理：在 W4A8 量化推理中，PTS（Power-of-Two Scaling）对激活施加通道级 2^δ 缩放，矩阵乘法中 2^δ 因子等价于对量化权重 W̃ 执行左移操作：W̃^{shifted}_{kj} = W̃_{kj} ≪ δ_k = W̃_{kj} × 2^{δ_k}。由于现代 GPU 没有原生的 "multiply-bitshift-add" 融合指令，直接在 multiply-accumulate 路径中插入 shift 会低效。DMQ 将 shift 操作放在权重加载阶段——权重从 packed INT4 解包后、进入 GEMM 累加前，在寄存器中完成位偏移。这样 shift 操作不进入 MAC 流水线，每 bit 的 shift 仅需约 1 个 cycle。DMQ 验证了该策略的实际效率：自定义 CUDA kernel 在 M=3072 时相比 PyTorch FP32 GEMM 达到 5.17× 加速，bit-shift 开销极小且被 GEMM 的 memory-bound 特性掩盖。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 DMQ 的 W4A8 GEMM kernel with bit-shift 为例：

```cuda
// 自定义 CUDA kernel: fused Quant + Bit-shift + GEMM + Dequant (W4A8)
// 输入: packed_W [C_out][C_in/2] INT4, X [B][C_in] INT8
//       s_X scalar, s_W [C_out], delta [C_in] (PTS exponents)
// 输出: Y [B][C_out] FP32

__global__ void gemm_w4a8_bitshift(
    const uint8_t* packed_w,   // INT4 packed weights
    const int8_t* input,        // INT8 quantized activations
    const int* delta,           // PTS shift amounts per channel [C_in]
    float* output,              // FP32 output
    int B, int C_in, int C_out,
    float s_x, const float* s_w
) {
    int row = blockIdx.x;  // output row (batch element)
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // output column
    
    if (col >= C_out) return;
    int accum = 0;  // INT32 accumulator
    
    for (int k = 0; k < C_in; k += 2) {
        // Step 1: Load packed INT4 weight byte
        uint8_t byte = packed_w[col * (C_in/2) + k/2];
        
        // Step 2: Unpack two 4-bit weights (sign-extend to INT8)
        int8_t w0 = (int8_t)((byte & 0x0F) << 4) >> 4;  // sign extend
        int8_t w1 = (int8_t)((byte >> 4) << 4) >> 4;
        
        // Step 3: BIT-SHIFT — apply PTS factor 2^{delta[k]}
        // Done in register, NOT in MAC path
        w0 = w0 << delta[k];    // = w0 * 2^{delta[k]}
        w1 = w1 << delta[k+1];
        
        // Step 4: INT8 MAC (fused into standard MAD instructions)
        accum += (int)w0 * (int)input[row * C_in + k];
        accum += (int)w1 * (int)input[row * C_in + k+1];
    }
    
    // Step 5: Dequantization
    output[row * C_out + col] = s_x * s_w[col] * (float)accum;
}
```

关键设计点：
- **Shift 位置**：在权重解包后、MAC 前（不在 MAC 内部），避免打破 GPU 的 MAD 指令融合
- **寄存器中完成**：shift 结果仅存在寄存器中，不写回 shared memory
- **选择性应用**：δ_k = 0 时 shift amount = 0，即 w << 0 = w（无操作）
- **每个 shift 仅 1 cycle**，对 GEMM 的整体 memory-bound 特性影响微乎其微

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bit-shift 在 GPU 上通过 PTX 指令 `shl.b32` 或 CUDA C 直接使用 `<<` 运算符实现。DMQ kernel 中 PTS 仅应用于 skip connection 层（网络总层数的 ~10-15%），整体延迟增加可忽略。Section E 的延迟测量：M=3072 时 kernel 延迟远低于 FP32 GEMM baseline，bit-shift 开销被 GEMM 主循环的 memory 开销完全覆盖。该技术也适用于其他 power-of-two 量化的硬件部署场景（如 ARM NEON 的 `vshlq_s32` 或 x86 AVX-512 的 `vpslld`），尤其适合边缘设备中乘法器受限的场景。

涉及论文标题：
- DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

---

## Multi-Kernel Mixed-Precision (MKMP) Multiplier

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Kernel Mixed-Precision (MKMP) Multiplier是Squat (EdgeQAT)论文提出的SIMD-based混合精度矩阵乘法器，用于在移动设备上高效执行sub-8-bit混合精度MAC操作。核心设计：将现有的INT8 multiplier与自定义INT4 multiplier整合到同一个GeMM kernel中，由Token Control Logic Module (TCLM)根据每个token的位宽动态路由到对应multiplier。INT4 multiplier基于INT8 multiplier构建，通过将相邻行4-bit权重拼接存入16-bit寄存器，利用ARM `mla`指令（32-bit目标寄存器INT32）在单条指令内完成乘加。理论上4-bit GEMM的计算操作数减半（vs 传统零扩展到8-bit）。MKMP multiplier解决了两个关键挑战：(1)标准SIMD INT8 kernel不支持混合精度；(2)sub-8-bit数据需零扩展至byte边界，浪费计算能力。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MKMP multiplier在移动CPU上的执行流程（以LLaMA-58M W4A8(1:1)推理一个token为例）：
```
// TCLM: Token分组
float scores[N];  // attn[:, 0]
int k = N * rho;
int threshold = heapsort_topk(scores, k);

// 分组8-bit和4-bit tokens
vector<float> x_8_group, x_4_group;
for (int i = 0; i < N; i++)
    if (scores[i] >= threshold) x_8_group.push_back(x[i]);
    else x_4_group.push_back(x[i]);

// === INT8 Multiplier (处理8-bit tokens) ===
// 标准ARM NEON vmlaq_s8() SIMD指令
// x8_int8: INT8 quantized activations
// w_packed: INT4 packed weights (offline quantized)
int8x16_t x8 = vld1q_s8(x8_int8_ptr);
int8x16_t w8 = vld1q_s8(w8_ptr);  // 4-bit权重先解包为8-bit
int32x4_t acc8 = vmlaq_s32(acc8, x8, w8);  // 乘加

// === INT4 Multiplier (处理4-bit tokens, INT4 Concatenation) ===
// Step 1: 加载相邻两行4-bit权重拼接为16-bit
// w_row_i: 4-bit, w_row_i+1: 4-bit -> concat: [w_row_i | w_row_i+1] in 16-bit
uint16x8_t w_concat = load_concat_4bit_weights(w_ptr);

// Step 2: 4-bit激活加载
uint8x16_t x4 = vld1q_u8(x4_int4_ptr);

// Step 3: 16-bit宽乘加 (ARM mla: 16-bit x 16-bit -> 32-bit acc)
// 内部拆分保持数学精度
int32x4_t acc4 = int4_concat_mma(x4, w_concat, acc4);

// Step 4: Bit-shift + row-wise summation
acc4 = vshlq_s32(acc4, shift_amounts);
acc4 = row_wise_sum(acc4);

// === 合并结果（反量化） ===
output_8 = dequantize(acc8, alpha_x8, alpha_w);
output_4 = dequantize(acc4, alpha_x4, alpha_w);
output = concat_and_reorder(output_8, output_4);  // 恢复原始token顺序
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MKMP multiplier的INT4部分基于gemmlowp/QNNPACK的INT8 GEMM kernel修改。INT4 concatenation利用ARM `mla`指令的32-bit目标寄存器特性（更宽的INT32累加器可容纳更多部分和而不溢出）。Compiler-level优化：分配计算线程时考虑不同操作的内存读取模式，重叠内存读取时间。移动CPU上实测加速：LLaMA-58M OnePlus 11 W4A4=2.24× vs FP16，GPT2-97M Raspberry Pi 5 W4A4=2.37× vs FP16。混合精度W4A8(1:1)在Raspberry Pi上额外加速超40%（vs pure W8A8）。代码开源：https://github.com/shawnricecake/squant。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

---

## INT4 Concatenation for SIMD Mobile Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT4 Concatenation是Squat论文在SIMD-based MKMP Multiplier中使用的技术，通过将两个4-bit权重值拼接存入单个16-bit寄存器，利用ARM CPU的`mla`指令（16-bit × 16-bit → 32-bit INT32累加器）在单指令内同时执行两个子字节乘加操作。传统方法将4-bit数据零扩展至8-bit（byte boundaries），浪费了一半的SIMD计算带宽。Concatenation技术将相邻行权重拼接后与共享激活值相乘，配合bit-shift和row-wise summation恢复正确结果。低比特优先策略（low-bit priority strategy）均匀利用位宽，最小化冗余零。该技术也可以推广到activation-activation的矩阵乘法中。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
INT4 Concatenation的寄存器级操作（ARM NEON）：
```
// 传统方法（baseline）：4-bit → 8-bit extension
// 每个4-bit权重占用8-bit寄存器空间，SIMD利用率50%
uint8x16_t w_4bit = vld1q_u8(w_ptr);  // 16个8-bit槽，只用了低4bit
// 浪费了16×4bit = 64bit的SIMD带宽

// INT4 Concatenation方法：
// w_row_i: [w_i0|w_i1|...|w_i7] (8个4-bit权重 in 32-bit)
// w_row_j: [w_j0|w_j1|...|w_j7] (相邻行, 8个4-bit权重 in 32-bit)
// concat: [w_i0|w_j0|w_i1|w_j1|...|w_i7|w_j7] (16个4-bit权重 in 64-bit)

// Step 1: 拼接加载
uint16x8_t w_concat;  // 8个16-bit槽 = 128-bit NEON寄存器
// w_concat[0] = (w_row_i[0] << 4) | w_row_j[0]  // low-bit priority
// w_concat[1] = (w_row_i[1] << 4) | w_row_j[1]
// ...

// Step 2: 激活值广播
uint16x8_t x_broadcast = vdupq_n_u16(x_shared);  // 同一激活值复制8份

// Step 3: 并行乘加 (mla: 16-bit × 16-bit → 32-bit累加)
int32x4_t acc_lo = vmull_s16(vget_low_s16(w_concat), vget_low_s16(x_broadcast));
int32x4_t acc_hi = vmlal_s16(acc_lo, vget_high_s16(w_concat), vget_high_s16(x_broadcast));

// Step 4: 内部拆分 (bit-shift恢复)
// acc中的每个32-bit值是 (w_i << 4 | w_j) * x_shared
// 需要shift来分离w_i*x_shared和w_j*x_shared
int32x4_t result_i = vshrq_n_s32(acc_hi, 4);  // 右移恢复高位结果
int32x4_t result_j = vandq_s32(acc_hi, mask_low4);  // mask提取低位结果
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
INT4 concatenation要求：(1) weight matrix的行数为偶数（相邻行配对拼接）；(2) 16-bit中间寄存器提供冗余（实际4-bit值远超需求但方便SIMD操作）；(3) low-bit priority策略确保位宽均匀利用。该技术理论上将4-bit GEMM的计算量减半（乘法和加法各减少50%），INT4 multiplier节省50% INT8 multiplier硬件资源。适用于所有支持16-bit乘法的ARM NEON处理器（ARMv7+）。也可应用于activation-activation矩阵乘法。限制：需确保16-bit乘法不溢出32-bit累加器（batch size或accumulation depth受限）。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

---

## Token Control Logic Module (TCLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Control Logic Module (TCLM) 是 Squat 论文中实现Token自适应量化的运行时控制模块。TCLM负责在推理（和训练）的每一步中：(1)根据最新注意力图的初始token列评估每个token的重要性；(2)通过Heapsort快速TopK选择指定比例的重要token；(3)将重要token（8-bit）和非重要token（4-bit）分别拼接分组；(4)调度对应的multiplier（INT8 multiplier和INT4 multiplier in MKMP）执行混合精度MAC。TCLM本身作为轻量级逻辑控制单元，其Heapsort和Concatenation开销可忽略（论文称"marginal overhead"）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TCLM在推理过程中的伪代码：
```
// TCLM: Token Control Logic Module
// 输入: attention_map [N, N], activations [N, D], rho (重要比例)
// 输出: 分组后的量化激活 + 位宽mask

struct TCLMOutput {
    int8_t* x_8bit_group;   // 8-bit量化后拼接的重要tokens
    int4_t* x_4bit_group;   // 4-bit量化后拼接的非重要tokens
    int* index_map;          // 结果排序的索引映射 [N]
    int n_8, n_4;            // 各组token数
};

TCLMOutput tclm_forward(float* attn, float* x, int N, int D, float rho) {
    // Step 1: 评估重要性
    float scores[N];
    for (int i = 0; i < N; i++)
        scores[i] = attn[i * N];  // attn[:, 0], 对初始token的注意力
    
    // Step 2: Heapsort TopK (O(N log k), k = rho*N)
    int k = (int)(rho * N);
    float threshold = heapsort_topk(scores, N, k);
    
    // Step 3: 分组 + 拼接
    vector<float> grp8, grp4;
    vector<int> idx8, idx4;
    for (int i = 0; i < N; i++) {
        if (scores[i] >= threshold) {
            grp8.insert(grp8.end(), &x[i*D], &x[(i+1)*D]);
            idx8.push_back(i);
        } else {
            grp4.insert(grp4.end(), &x[i*D], &x[(i+1)*D]);
            idx4.push_back(i);
        }
    }
    
    out.x_8bit_group = layer_wise_quant8(grp8.data(), grp8.size());
    out.x_4bit_group = layer_wise_quant4(grp4.data(), grp4.size());
    out.n_8 = idx8.size(); out.n_4 = idx4.size();
    // index_map记录原始顺序，用于合并时恢复
    merge_index_map(out.index_map, idx8, idx4);
    return out;
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TCLM实现要点：(1)Heapsort TopK是标准C++算法库函数，开销O(N log k) vs 全排序O(N log N)；(2)分组拼接操作可通过memcpy或指针重排实现，不涉及计算；(3)在MKMP multiplier中，TCLM的输出直接路由到对应的INT8或INT4 kernel，kernel间无缝衔接。训练时每步调用TCLM（基于当前注意力图），推理时同样动态执行。TCLM本身无训练参数，仅做逻辑控制。Squat中的TCLM实现：代码开源https://github.com/shawnricecake/squant。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

## BitBLAS

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BitBLAS 是 Microsoft 开发的高性能低比特 LLM 推理算子库，为 INT4/INT3/INT2 等低比特量化模型提供 GPU kernel 实现。全称 "Bit-BLAS"（结合 bit-level 操作和 BLAS 接口设计）。BitBLAS 是 Ladder 项目（OSDI 2024）的子项目，专注于低精度张量运算的硬件感知优化。在 EfficientQAT 中使用 BitBLAS 评估量化模型的实际推理加速——通过将 FP16 矩阵向量乘法替换为 BitBLAS 的 INT2 kernel，获得 2.9x-4.4x 的前向加速。BitBLAS 通过对不同位宽自动生成优化的 CUDA kernel（利用 Tensor Core 或 CUDA Core），将低比特打包权重直接送入硬件算术单元，避免运行时解量化开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
BitBLAS 的 INT2 矩阵向量乘法（Matrix-Vector Multiplication, GEMV）kernel 执行流程：
```
# BitBLAS INT2 GEMV kernel (简化)
KERNEL int2_gemv(Weight_packed_int2, Input_fp16, Output_fp16):
    smem_w = load_and_depack(Weight_packed_int2, tile_id)  # 解包INT2 weight tile到共享内存
    smem_x = load_input_tile(Input_fp16, tile_id)
    accum = 0
    for k in range(K / TILE_K):
        w_tile = smem_w[:, k*k_step : (k+1)*k_step]       # INT2 weight tile
        x_tile = smem_x[k*k_step : (k+1)*k_step]           # FP16 input tile
        # 低精度MAC + 反量化缩放
        mac_partial = int_mad(w_tile, x_tile)               # INT2 * FP16 → FP32累加
        accum += mac_partial * scale + zero_point_adjust    # 反量化
    output[tile_id] = accum
    return
```
EfficientQAT 在 A100-80GB 上测试的 BitBLAS INT2 加速比（Table 10）：Llama-2-7B size=4096x4096: 3.1x, 11008x4096: 2.9x; Llama-2-13B: 3.6x/3.5x; Llama-2-70B: 3.9x/4.4x。加速比随矩阵尺寸增大而提高，因解包开销在更大矩阵中被摊销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BitBLAS 通过硬件感知的张量变换（Tensor Transformation）实现低比特加速：核心策略是将不同位宽的量化权重通过 ladder 变换映射到 GPU 硬件原生支持的数据格式和指令（如利用 Tensor Core 的 INT8 mma 指令模拟 INT2/INT4 计算，或使用 CUDA Core 的 bit-serial 执行）。使用方式：(1) 环境安装：`pip install bitblas`；(2) 模型集成：替换 HuggingFace 模型的 Linear 层为 BitBLAS 低比特算子；(3) 代码调用：`bitblas.matmul(weight_packed, input, bit=N, group_size=g)`。BitBLAS 支持 INT2/INT3/INT4 等多种位宽，与 GPTQ、AWQ、EfficientQAT 等量化方法的输出格式兼容。可替代 MLC-LLM 和 Marlin kernel 作为低比特推理后端。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models

---

## TBL (Table Lookup) Instruction for VQ Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TBL（Table Lookup）指令是移动 CPU（ARM 架构）的 SIMD 硬件指令，用于将索引值映射到查找表中的预存值。典型规范：5-6 bit 索引 → 8-bit 值（如 ARM NEON 的 `vtbl`/`vtbx` 指令系列）。在 GPTVQ 中，TBL 指令是 VQ 解码的核心算子——每个 VQ 维度需要一个 TBL 调用将 6-bit 质心索引映射到 8-bit signed integer 值。2D VQ 需要 2 条 TBL 指令（每维一条），结果相加后乘以 scale。TBL 指令将 16 个 8-bit 表项存于 128-bit NEON 寄存器中，单周期完成 16 路并行查表（one register = 16 × 8-bit = 128 bits，正好一个 64-entry codebook 需要 4 个 NEON 寄存器）。相比通用 gather/scatter 指令（如 SVE gather），TBL 指令延迟更低、吞吐更高，是移动端高效 VQ 解码的硬件基础。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// ARM NEON TBL-based 2D VQ Decode Kernel (伪代码)
// 输入: packed 6-bit indices, 64-entry LUT (8-bit signed), FP16 scale
// 输出: dequantized FP16 weights

// NEON 寄存器布局:
// v0-v3: 4 × 128-bit = 64 × 8-bit LUT entries
// v4: packed 6-bit indices (16 indices per cycle)

void vq_decode_2d_neon(
    const uint8_t* packed_indices,  // 6-bit packed
    const int8_t* lut_dim0,         // 64 entries × 8-bit
    const int8_t* lut_dim1,         // 64 entries × 8-bit
    const float16_t* scale,         // per-block FP16 scale
    float16_t* decoded_weights,     // output
    int num_weights
) {
    // 加载 LUT 到 NEON 寄存器 (64 entries = 4 × 128-bit regs)
    int8x16_t lut0_0 = vld1q_s8(lut_dim0);       // entries 0-15
    int8x16_t lut0_1 = vld1q_s8(lut_dim0 + 16);  // entries 16-31
    int8x16_t lut0_2 = vld1q_s8(lut_dim0 + 32);  // entries 32-47
    int8x16_t lut0_3 = vld1q_s8(lut_dim0 + 48);  // entries 48-63
    
    // 同理加载 dim1 LUT
    
    for (int i = 0; i < num_weights; i += 16) {
        // Step 1: 从 packed format 解包 16 个 6-bit indices
        uint8x16_t indices_packed = vld1q_u8(packed_indices + i*6/8);
        uint8x16_t idx = unpack_6bit(indices_packed);  // 解包
        
        // Step 2: TBL 查表（维度 0）
        // vtbl: 用 idx 的低 4-bit 选择寄存器，高 2-bit 选择表段
        int8x16_t val_dim0 = vtbx4_s8(
            vtbx4_s8(vtbl4_s8(lut0_0, idx_low), lut0_1, idx_low),
            lut0_2, lut0_3, idx_low
        );
        
        // Step 3: TBL 查表（维度 1）
        int8x16_t val_dim1 = /* 同理用 lut_dim1 查表 */;
        
        // Step 4: 合并两维 + 反量化
        int8x16_t val_sum = vaddq_s8(val_dim0, val_dim1);  // v1 + v2
        float16x8_t decoded = vcvtq_f16_s16(             // int8 → float16
            vmovl_s8(vget_low_s8(val_sum))
        );
        decoded = vmulq_f16(decoded, vdupq_n_f16(*scale));  // × scale
        
        vst1q_f16(decoded_weights + i, decoded);
    }
}
```

关键设计决策：
- 6-bit index 限制 codebook ≤ 64 entries（= 2^6），精确匹配 TBL 指令的寻址能力
- 2D VQ 每 weight 需 2 次 TBL 查表 + 1 次加法 + 1 次乘法
- Packed 6-bit 格式：16 个 weights × 6 bits = 96 bits，占 12 bytes（vs 16 × 8 bits = 16 bytes）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TBL 指令在 ARM NEON 中为 `vtbl`（索引在 0..n-1 范围内返回表值，超出返回 0）和 `vtbx`（索引超出范围保留目标寄存器原值）。x86 等效指令为 `pshufb`（SSSE3+）。在 GPTVQ 中，TBL 是推理引擎 C 代码中通过 NEON intrinsics（`vld1q_s8`, `vqtbl1q_s8` 等）调用的。Codebook 存为 INT8 格式（8-bit signed），对应 TBL 的 8-bit 输出。关键限制：(1) TBL 仅支持 5-6 bit index（取决于实现），因此 VQ codebook 必须 ≤ 64 entries；(2) 解包 6-bit indices 的 overhead 是 TBL 之外的额外开销；(3) TBL 的 128-bit 限制意味着更大 codebook 需要多次 TBL 调用。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---

## VQ Decode Kernel on Mobile CPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VQ Decode Kernel on Mobile CPU 是 GPTVQ 推理引擎中负责将 VQ 压缩权重在线解码为 native compute data type 的软件 kernel。它的设计目标是解码延迟低于 DRAM 带宽，使 VQ footprint 减小转化为实际的 token rate 提升。kernel 流程：从 DRAM 加载 block tuple（packed 6-bit indices + 64-entry INT8 LUT + FP16 scale）→ 进入 CPU cache → 解包 6-bit indices → TBL 指令查表（每维一次）→ 合并两维 → scale 反量化 → SIMD GEMM。在 Snapdragon X Elite 上实测：VQ 2D 3.125 bpv 解码延迟 = 0.96× vs INT4 数据传输（Table 6），端到端 token rate 26.15 tok/s（+10% vs Ours INT4, +45.7% vs llama.cpp INT4），footprint 3.52GB（-19% vs INT4 4.33GB）。

在 NVIDIA GPU（RTX 3080）上也实现了 VQ decode kernel，使用 CUDA vector types（char4/uchar4 和自定义 char128 agglomeration）并行加载和解码。VQ 4D 2.125 bpv 在 GPU 上实现相对 footprint 0.53× + 相对延迟 0.71×（vs INT4），即同时减小 footprint 和延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU 端 VQ decode + GEMM 完整调度流程：

```
Sequence Diagram: CPU VQ Decode + GEMM (per Transformer layer)
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  DRAM   │     │ CPU Cache│     │ NEON Regs│     │ SIMD ALU │
└────┬────┘     └─────┬────┘     └─────┬────┘     └─────┬────┘
     │                │                │                │
     │ Block Load ───>│                │                │
     │ (indices+LUT   │                │                │
     │  + scale)      │                │                │
     │                │ TBL LUT ──────>│                │
     │                │ load (dim0)    │                │
     │                │                │ TBL dim0 ─────>│
     │                │                │ (6b→8b)        │
     │                │                │<───────────────│
     │                │ TBL LUT ──────>│                │
     │                │ load (dim1)    │                │
     │                │                │ TBL dim1 ─────>│
     │                │                │ (6b→8b)        │
     │                │                │<───────────────│
     │                │                │ ADD dim0+dim1─>│
     │                │                │<───────────────│
     │                │                │ MUL scale ────>│
     │                │                │ (int→fp16)     │
     │                │                │<───────────────│
     │                │                │                │
     │                │                │ SIMD GEMM ────>│
     │                │                │ (fp16 weights  │
     │                │                │  × fp16 act)   │
     │                │                │<───────────────│
     │                │                │                │
     │<───────────────│ output write   │                │
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 的推理引擎是 Qualcomm 自研的 C 语言实现：(1) 使用 ARM NEON intrinsics 实现 TBL 查表和 SIMD GEMM；(2) 利用 polyhedral compiler（Polly）进行细粒度循环向量化优化；(3) 粗粒度并行利用 transformer 的结构特性（如 multi-head attention 的 head 间并行）。移动端部署在 Snapdragon X Elite（Windows + Clang 18.1）。GPU kernel 在 CUDA 上实现，针对 RTX 3080+ 验证。代码尚未开源（论文声明 "will be made available in the future"）。关键设计原则：解码延迟必须 < DRAM 带宽节省的延迟，即 (T_decode - T_saved_bandwidth) < 0。GPTVQ 的 2D VQ 配置（6-bit index, 64-entry codebook）经过与 TBL 指令协同设计，确保此条件成立。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---

## Any-Precision-LLM Kernel（任意精度 LLM 推理核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Any-Precision-LLM 是 SNU 开发的 GPU CUDA kernel，用于 non-uniform scalar quantized LLM 的高效推理（支持 2/3/4-bit 多精度混合部署）。核心流程：(1) Bit-transpose：将 p 个独立 1-bit packed weight planes 重组为 p-bit centroid index；(2) Table lookup：用 index 查 per-channel codebook 恢复 FP16 权重；(3) FP16 GEMM（via cuBLAS）。每个 output channel 维护独立 codebook，支持混合 bit-width 模型在单 GPU 上运行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入：packed BP_0..BP_{p-1} (1-bit/plane), activation A (FP16), codebook C (2^p × FP16)
# 输出：O = A @ W_deq (FP16)

for each tile (m_tile, k_tile):
    # Step 1: Bit-transpose — 最大开销（35-58% kernel 延迟）
    idx[m,k] = Σ_{i=0}^{p-1} extract_bit(BP_i, m, k) << i
    
    # Step 2: Codebook lookup（9-17% kernel 延迟）
    w_deq[m,k] = C[idx[m,k]]
    
    # Step 3: cuBLAS GEMM
    O_tile += A_tile @ w_deq_tile
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：github.com/yeonhongpark/Any-Precision-LLM。GuidedQuant (ICML 2025) 在 end-to-end throughput benchmark 中使用此 kernel 测量 LNQ/LNQ+GuidedQuant 的非均匀标量推理吞吐（RTX 4090: 2-bit Llama-2-7B 347 tok/s）。局限性：bit-transpose 是 kernel 延迟主要瓶颈（AnyBCQ Table 7），后续工作 Quantix 通过 hardware-aligned bit shuffling 消除此开销。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---

## QTIP Kernel (HYB Variant / QTIP 向量量化推理核)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QTIP (Quantization with Trellises and Incoherence Processing, NeurIPS 2024) 是一种 weight-only vector quantization PTQ 方法，其 HYB (hybrid) 变体提供了开源 CUDA kernel 用于 GPU 推理。与 scalar quantization（每权重独立 index）不同，QTIP 将 d 个连续权重编码为一个 lattice codebook vector index，通过 trellis 结构化搜索优化量化。HYB kernel 使用小型的 LUT（适合 GPU L1 cache）进行向量解码，然后将解码后的 FP16 权重与 activation 进行 GEMM。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QTIP HYB kernel 的高层次流程：
```
# 输入：packed vector indices I (B bits per d-dim group)
#       activation A, lattice codebook L (shared or per-group)
# 输出：FP16 output

for each vector group g (size d):
    idx = unpack(I, g)                         # 解码 B-bit index
    w_vec = decode_lattice(L, idx)             # d 维向量（比 scalar 查表更复杂）
    for each output m:
        output[m] += dot(A[m, g_start:g_end], w_vec)
```
解码延迟 overhead 使 vector quantization 推理吞吐低于 scalar：RTX 4090 上 Llama-2-7B 2-bit non-uniform scalar 347 tok/s vs vector (QTIP HYB) 200 tok/s，即使 fusing Q/K/V projections 后 vector 仅达 248 tok/s（Table 2, Table 7）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：github.com/Cornell-RelaxML/qtip（HYB CUDA kernel）。QTIP 有三个变体：1MAD、3INST（LUT-free, 论文未开源 kernel）和 HYB（带 L1-cache 友好 LUT, 开源 kernel）。GuidedQuant 使用 QTIP HYB kernel 进行向量量化吞吐评估。推理吞吐结论：non-uniform scalar + Any-Precision-LLM kernel 在 memory-bound（batch=1）场景下比 vector + QTIP HYB kernel 提供更好的 latency-accuracy tradeoff。

在 QTIP 原始论文中的实现：bitshift trellis 解码无需索引解包和格点解码——每个权重通过 compute-based code (1MAD/3INST/HYB) 直接从 L-bit 状态字即时生成。流程为：读取 32-bit word → bitshift 获取 L=16 bit 状态 → 1MAD: (LCG + 4×8bit sum + scale) 2 instr / 3INST: (LCG + XOR FP16 + FADD) 3 instr / HYB: (hash + LUT lookup + sign flip) 摊销 2 instr → 输出 FP16 权重。16×16 tile 映射到 MMA tile 直接矩阵乘。QTIP 在 RTX 6000 Ada 上 Llama 2 7B 2-bit 达 188 tok/s (>3× FP16 55.9 tok/s)，与 QuIP# 吞吐持平（186 tok/s），但有效维度为 256（vs QuIP# 8D），量化质量更高而无额外推理开销。关键优化：HYB codebook 仅 2KiB（比 AQLM 1MiB 小 512×），可 32× 复制消除 bank conflicts；bitshift 解码完全并行化（对比 naive TCQ 的顺序依赖）。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

## Fused Selective Scan Kernel (Hardware-Aware SSM Scan)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Selective Scan 是 Mamba (Gu & Dao, 2023) 为选择性 SSM (S6) 设计的硬件感知 CUDA kernel，通过 kernel fusion + parallel scan + recomputation 使 time-varying SSM 在现代 GPU 上高效运行。核心问题：选择性 SSM 不再是 LTI（线性时不变），无法用卷积（FFT）模式；朴素循环需物化 (B,L,D,N) 中间状态 h（N=16 时比输入大 16 倍），HBM IO 量巨大。该 kernel 将离散化、parallel scan、输出乘加融合在 GPU SRAM 内，仅将 O(BLD) 最终输出写回 HBM，IO 减少约 N 倍。在 A100 GPU 上，该 kernel 在序列长度 >2K 后超越 FlashAttention-2，32K 时快约 7×；vs PyTorch naive scan 快 20–40×。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Fused Selective Scan Kernel (per chunk, in SRAM):

// Phase 1: Load from HBM to SRAM
Δ [B,L,D], A [D,N], B [B,L,N], C [B,L,N], x [B,L,D]

// Phase 2: Discretize in SRAM (per timestep, fused with scan)
for t in 0..L-1:
    Ā_t = exp(Δ_t ⊙ A)     // (D,N), element-wise vector*scalar exp
    B̄_t = Δ_t ⊙ B_t        // (D,N), 一阶近似: (exp(ΔA)-I)/(ΔA) ≈ I for small Δ

// Phase 3: Parallel Associative Scan (Blelloch) in SRAM
// element = (a: R^{N}, b: R^{N})
// binop: (a,b) ⊕ (a',b') = (a'⊙a, a'⊙b + b')
// Up-sweep → Down-sweep → outputs h_{0..L-1} in O(log L) parallel steps

// Phase 4: Output multiply in SRAM
y_t = C_t ⊙ h_t for t in 0..L-1  // (D,)

// Phase 5: Write y [B,L,D] to HBM (ONLY this goes to HBM)

HBM IO: Read O(BLD) + Write O(BLD) = O(2BLD)
Naive IO: Read O(3BLDN) + Write O(BLDN) + const = O(4BLDN)
```
反向传播采用重计算：不保存 h [B,L,D,N] → backward 重新加载 O(BLD) 输入到 SRAM → 重计算 h → 计算梯度 → 写回 O(BLD)。总 backward IO = O(BLD) vs 保存方案 O(BLDN)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/state-spaces/mamba (CUDA C++)。利用 GPU memory hierarchy: HBM→L2→SRAM→register。长序列分 chunk 处理，chunk 间通过 HBM 传递 scan state 连接。Mamba-125M activation memory ≈ 4.8-38.2GB (batch 1-32)，与 FlashAttention-2 Transformer (4.6-34.5GB) 可比。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Parallel Associative Scan (Blelloch Scan) for Recurrent Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parallel Associative Scan (Blelloch scan) 是将顺序递推 h_t = f(h_{t-1}, x_t) 并行化为 O(log L) depth 的算法，前提是 f 满足结合律。在 Mamba SSM 中，递归 h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_t 的关联操作为 elem=(a,b), binop: (a,b) ⊕ (a',b') = (a'⊙a, a'⊙b + b')。S5 (Smith et al., 2023) 首次将 parallel scan 用于 SSM，但需切换为 MIMO 降低 state 维度。Mamba 保持 SISO 高 state 维度 + 硬件感知实现克服计算问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Blelloch Scan for SSM: elem=(a,b), ⊕=(a'·a, a'·b+b')

// Up-Sweep (Reduce): combine adjacent pairs
for d = 0 to log₂(L)-1:
    stride = 2^{d+1}, gap = 2^d
    for k in 0..L/stride-1 (parallel threads):
        left = k*stride + gap - 1
        right = k*stride + stride - 1
        (a_r, b_r) = data[right]
        (a_l, b_l) = data[left]
        data[right] = (a_r·a_l, a_r·b_l + b_r)  // combine

// Down-Sweep (Distribution): propagate prefix
data[L-1] = (1, 0)  // identity element
for d = log₂(L)-1 down to 0:
    stride = 2^{d+1}, gap = 2^d
    for k in 0..L/stride-1 (parallel threads):
        left = k*stride + gap - 1
        right = k*stride + stride - 1
        tmp = data[left]
        data[left] = data[right]
        data[right] = data[right] ⊕ tmp

// Output: data[t] = h_t for all t
```
在 GPU 上以 tile 为粒度实现：每 SM 处理 chunk 内 scan → 跨 SM partial scan → 组合。Mamba 的 fused kernel 将 scan 完全限定在 SRAM 完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
适用于任何满足结合律的递推关系（RNN/GRU/LSTM 等 gate-based RNN）。Work = O(L), Depth = O(log L)。在 Mamba 中与离散化和输出乘加融合为单一 kernel。关键限制：需要 L 对齐到 2 的幂（pad if needed）；仅支持关联操作（不要求交换律）。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Recomputation (Rematerialization) in SSM Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Recomputation（梯度重计算/rematerialization）是内存优化技术：前向不保存大中间激活，反向时重新计算。在 Mamba 中，中间状态 h ∈ R^{B,L,D,N}（N=16 比输入大 16 倍）若保存到 HBM 内存开销巨大。解决：前向仅保存 O(BLD) 输入 → 反向重新加载到 SRAM → 重计算 h → 计算梯度。因输入+梯度 O(BLD) 远小于 h O(BLDN)，总 HBM IO 反而更少（memory-bandwidth 是瓶颈）。总效果：每个 selective SSM 层 ≈ 16 bytes/token 激活内存 vs Transformer (FlashAttention+MLP) ≈ 32 bytes/token。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Forward: don't save h (O(BLDN)), save only inputs (O(BLD))
fused_scan_forward(x, Δ, A, B, C) → y

// Backward: reload inputs, recompute h, compute gradients
fused_scan_backward(dL/dy, x, Δ, A, B, C):
    // 1. Load x, Δ, A, B, C from HBM → SRAM  (O(BLD) read)
    // 2. Recompute h in SRAM:
    //    discretize(Δ, A, B) → Ā, B̄
    //    parallel_scan(Ā, B̄⊙x) → h
    // 3. Compute gradients using h and dL/dy:
    //    dL/dC = dL/dy ⊙ h  → scan backprop → dL/dΔ, dL/dB
    // 4. Write gradients to HBM  (O(BLD) write)

IO对比:
  重计算: Read O(BLD) + Write O(BLD) = O(2BLD)
  保存: Read h O(BLDN) + Read grad O(BLD) = O(BLD(N+1))
  N=16: 重计算 IO ≈ 2BLD vs 保存 IO ≈ 17BLD → 8.5× 节约
```
该技术与 FlashAttention 的重计算策略一致（不保存中间 attention matrix，反向重计算 softmax）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源: https://github.com/state-spaces/mamba (fused scan kernel 内实现)。Mamba 还将重计算扩展到整个 SSM block：不保存 activation 输出和 short convolution 中间结果，需要时快速重计算。适用条件：重计算开销 < 额外 HBM IO 开销，对 memory-bandwidth-bound 操作（scan, attention）通常成立。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Fused Reorder-and-Quantize Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Reorder-and-Quantize Kernel（融合重排量化kernel）是 MicroMix 的 GPU kernel 优化技术，将激活张量的通道重排（channel reordering）和 block-wise MX 量化融合为单个 CUDA kernel。混合精度 MX 量化中，相邻通道可能分配不同精度（MXFP4/6/8），需将同精度通道重排到连续块以实现规则内存访问。融合消除中间结果的 global memory 往返，将重排+量化开销控制在总 kernel 时间 20% 以内。

从kernel调度角度拆解术语，给出伪代码或具体计算过程。
```
__global__ void fused_reorder_quantize(
    half* X, int* sigma, int p4_K, int p6_K,
    MXFP4* out_G4, MXFP6* out_G6, MXFP8* out_G8
) {
    __shared__ half tile[BLOCK_M][BLOCK_K];
    load_tile(X, tile);  // coalesced read from global memory
    
    // 在 shared memory 中按 sigma 重排列
    __shared__ half reordered[BLOCK_M][BLOCK_K];
    for (int j = 0; j < BLOCK_K; j++)
        reordered[threadIdx.y][j] = tile[threadIdx.y][sigma[block_start + j]];
    
    // 分组量化：G4 (MXFP4) / G6 (MXFP6) / G8 (MXFP8)
    quantize_mxfp4_block(reordered[:, 0:p4_K], out_G4);
    quantize_mxfp6_block(reordered[:, p4_K:p4_K+p6_K], out_G6);
    quantize_mxfp8_block(reordered[:, p4_K+p6_K:K], out_G8);
}
// 每个 quantize_mxfpX_block 内部对 32 元素 block:
//   scale = 2^{floor(log2(max_abs)) - b}; Q(x) = round(clip(x/scale, -q_max, q_max))
```
注解：输入 X 为 FP16 [M, K]；sigma 为离线预计算的通道排列索引；p4_K/p6_K 为整数分界；输出三组 MX 张量直接供后续 MXFP GEMM 使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MicroMix 基于 CUDA 实现，shared memory 作重排缓冲区。要点：(1) coalesced 加载 FP16 激活；(2) shared memory 列索引重映射避免 global memory irregular access；(3) 量化紧接重排，直接输出 MX 格式；(4) 输出直连后续三路 MXFP GEMM kernel。RTX 5090 上 fused reorder-and-quantize 仅占总 kernel 时间 7.9%-17.0%（seqlen 128→4096），GEMM 占 83.0%-92.1%。适用于任何需通道重排+量化连续执行的混合精度量化场景。

涉及论文标题：
- MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models

---

## Group-GEMM（分组通用矩阵乘法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group-GEMM（Group General Matrix Multiply）是并行执行多个形状不同但独立的 GEMM 操作的 GPU 计算模式。在 MoE 模型中，每个 token 的路由机制选择 top-k 个 expert，不同 expert 收到的 token 数不同，因此 per-expert GEMM 的形状（m 维度 = token 数）不同。Group-GEMM 将所有这些 shape 不同的 GEMM 打包为单次 kernel launch 并行执行。与 Batched-GEMM（所有子问题形状完全相同）不同，Group-GEMM 处理的是异构 shape 子问题，需要更精细的 tile 分解和调度策略。NVIDIA CUTLASS 提供高效的 Group-GEMM 实现。

在 MxMoE 中，Group-GEMM 被扩展为支持混合精度：同一 kernel launch 内不同 expert 的 GEMM 可以使用不同的精度（如 W4A16, W8A8, W4A4），进一步增加了 tile shape 和计算模式的异构性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE block 中 Group-GEMM 的计算结构：

```
MoE block 包含 E 个 expert，每个 expert 3 个 linear block (gate/up/down)

给定输入 X ∈ R^{T×d} (T tokens):
1. Gating: 每个 token 分配给 top-k expert
   → per-expert token 数 T_e，Σ T_e ≤ T×k

2. Group-GEMM 并行执行:
   for expert e in activated experts:
       X_e = gather(X, tokens_assigned_to_e)  // [T_e, d]
       # 3 个 GEMM 可进一步融合或分开发射
       gate_e = X_e × W_gate_e^T  // [T_e, d] × [d, d_inter]
       up_e   = X_e × W_up_e^T    // [T_e, d] × [d, d_inter]
       down_e = (SiLU(gate_e) ⊙ up_e) × W_down_e^T  // [T_e, d_inter] × [d_inter, d]
   
   # 所有 expert 的 GEMM 打包为 Group-GEMM 并行执行
   # 不同于顺序执行 (for-loop)，CUTLASS Group-GEMM
   # 在单 kernel 内将所有 tile 调度到 SM 上并行处理

3. Final output:
   F = Σ_e w_e · scatter(down_e)  // 加权聚合回 token 顺序
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 使用 `cutlass::gemm::kernel::GroupedGemm` 和 `cutlass::gemm::device::GroupedGemm` 模板类实现。MxMoE 在此基础上扩展：每种精度实现独立的 micro-kernel（CTA 级 CUDA device function），由 kernel generator 自动组合不同 micro-kernel 为统一 fused kernel，tile scheduler 按 greedy LPT 调度 tile 到 SM。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---

## Micro-Kernel Specialization for Mixed-Precision GEMM（混合精度 GEMM 的微内核特化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Micro-Kernel Specialization 是 MxMoE 提出的 CTA 级 CUDA kernel 设计策略：为每种量化精度（如 W2A16, W4A16, W4A4-g128, W8A8）实现专用的 micro-kernel，而非开发一个 universal kernel 处理所有精度。每个 micro-kernel 是 CTA index-independent 的 CUDA device function，资源通过 C++ template 参数指定，memory access 针对该精度的计算-访存模式手工调优。例如：W2A16 micro-kernel 集成 fused dequantization + bit manipulation 优化 int-to-float 转换；W4A4-g128 micro-kernel 使用 multistage software pipelining 严格遵循 128 量化 group 约束。

该策略的关键优势：对比 universal kernel（所有精度共享同一代码路径），specialized micro-kernel 消除了运行时条件检查（避免阻碍 MAC-loop 展开），允许针对精度特性选择最优 tile size。对比手工为每种精度组合写 kernel（|S|! 个），specialized micro-kernels 只需实现 |S| 个可配置 micro-kernel，由 kernel generator 自动组合。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Micro-kernel 对比示例（W4A4）：

```
Universal Kernel (单 kernel 处理所有精度):
  // 运行时条件判断
  if (group_size == 128):
      // 受限 tile k 选择 (per-channel kernel tile k=256 不可用)
      load_and_dequant_group128(...)
  else:
      load_and_dequant_per_channel(...)
  # 性能: W4A4 per-channel 929 TOPS, W4A4 group128 412 TOPS

Specialized Micro-Kernel (MxMoE):
  // W4A4 per-channel micro-kernel
  template<int TileM, int TileN, int TileK>
  __device__ void w4a4_per_channel_micro_kernel(...) {
      // 专为 per-channel 优化的 dequant + MMA pipeline
      // 无运行时分支, 全循环展开
  }
  # 性能: W4A4 per-channel 1070 TOPS, W4A4 group128 667 TOPS
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MxMoE 的 kernel generator 根据 ILP 分配的方案，从 micro-kernel 库中选择对应的 micro-kernels，通过 resource configuration 统一 warp count 和 shared memory 后，编译为单个 fused mixed-precision Group-GEMM kernel。适用于任何需要混合精度并行执行的场景。推广到更多精度仅需增加 micro-kernel 实现（O(|S|)），而非手工枚举所有组合（O(|S|!)）。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---

## Tile Scheduling via Greedy LPT（基于贪婪 LPT 的 Tile 调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tile Scheduling via Greedy LPT 是 MxMoE 用于将混合精度 Group-GEMM 的 tile 分配到 GPU SM 的调度策略。这是经典的 makespan minimization 问题（P||C_max）：给定 M 个 SM 和 N 个 tile（每个 tile 的预估执行时间 c_t 不同），目标是最小化最慢 SM 的完成时间。MxMoE 使用 LPT（Longest Processing Time first）greedy 启发式：按 tile 执行时间 c_t 降序排列，依次将每个 tile 分配给当前累积负载最小的 SM。Graham (1966) 证明 LPT 在 tile 数远大于 SM 数时实现近最优性能（makespan ≤ (4/3 - 1/(3P)) × OPT），且调度开销远低于动态规划精确解。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: tiles = [(c_1, SM_id=None), ..., (c_N, SM_id=None)]
      P = number of SMs

LPT 调度:
  sort tiles by c_t descending
  SM_load = [0] * P  // 每个 SM 的累积负载

  for tile (c_t, _) in tiles:
      # 找到当前负载最小的 SM
      min_sm = argmin(SM_load)
      assign tile to SM min_sm
      SM_load[min_sm] += c_t

  makespan = max(SM_load)
  # 接近最优: ≤ (4/3 - 1/(3P)) × OPT
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MxMoE 运行时，tile scheduler 在 kernel launch 前执行：收集所有 expert 的 tile 列表 → greedy LPT 分配 → 编译进 kernel grid 配置。tile 数量通常远大于 SM 数（Qwen1.5-MoE 的 60+ expert 可产生数千 tile vs RTX 4090 的 128 SM），因此 LPT 近最优。调度开销 O(N log N)（排序），远低于 DP 的 O(N P^N)。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---

## K-Dimension Tiling / Slice-K（K 维度分块）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
K-Dimension Tiling（也称 Slice-K）是 GPU GEMM 中将矩阵乘法的 inner dimension（K 维度）进一步切分为多个子 tile 以增加并行度、提升 shared memory 利用率的技术。在 MxMoE 的混合精度 Group-GEMM 中，不同精度的 micro-kernel 天然使用不同的 tile size：例如 W4A16 的 tile 比 W8A8 的 tile 显著更小。当这些 micro-kernel 水平融合时，必须统一 shared memory 分配（取最大值），导致小 tile 的 micro-kernel 出现 shared memory under-utilization。Slice-K 解决方案：对 W4A16 配置额外沿 K 维度切分 tile——将单个大 K-tile 分为多个子 K-tile，每个子 K-tile 增加 warp utilization 并更好利用分配的 shared memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
标准 Tile (无 Slice-K):
  W4A16 tile: [M=128, N=64, K=256] → 使用 shared memory = 128*64*2B = 16KB
  W8A8 tile:  [M=128, N=64, K=256] → 使用 shared memory = 128*64*4B = 32KB
  统一分配 32KB shared memory → W4A16 浪费 16KB

With Slice-K (K 切分为 2):
  W4A16 tile: [M=128, N=64, K=128] × 2 个子 tile
  每个子 tile:
    - shared memory = 128*64*2B = 16KB (×2 子 tile = 32KB 充分利用)
    - warp utilization 提升 (2 倍子 tile 并行计算)
    - 子 tile 结果累加得到最终输出
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Slice-K 通过调整 tile configuration y_{i,j,k,t} 实现。在 ILP 求解阶段，tile 配置已包含 K 切分选项，ILP 自动选择最优的分片数和 tile shape 组合。CUTLASS 原生不支持 slice-K 与混合精度的联合优化，MxMoE 通过 kernel generator 自动注入 K-splitting logic。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

## INT4 2:4 Sparse GEMM (INT4 2:4 稀疏矩阵乘)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT4 2:4 Sparse GEMM 是将 INT4 量化和 2:4 结构化稀疏结合到同一个矩阵乘法 kernel 中的技术。权重 W 使用 INT4 量化（packed：每 2 个 4-bit 值打包为 1 byte），同时施加 2:4 结构化稀疏（每 4 个连续元素中恰好 2 个非零，50% 稀疏率）。激活 X 使用 INT4 量化。NVIDIA Ampere 架构起（SM 8.0+），Sparse Tensor Cores 原生支持 2:4 sparse MMA，结合 INT8/INT4 Tensor Core 可在单条指令中同时处理稀疏和低精度计算。相比 FP16 dense GEMM：4.72−5.9× 加速，6.4× 内存减少。相比 INT4 dense GEMM：1.4× 额外加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
INT4 2:4 Sparse GEMM 的 Kernel 执行流程（基于 CUTLASS 实现）：

```
// 输入: INT4 packed W^ (M×K, 每2值1byte), 2:4 metadata (M×K/4×2bit)
//       INT4 packed A^ (K×N), scale_w, scale_a
// 输出: FP32 C (M×N)
// GPU: NVIDIA A100 (Ampere, SM 8.0), CUTLASS API

// ===== 1. Global → Shared Memory (coalesced) =====
__syncthreads();
// 每个 threadblock 加载 tile
W_tile_int4  = load_global_to_shared(W^_tile)     // INT4 packed, 50% size
W_meta_tile  = load_global_to_shared(metadata_tile) // 2:4 selection indices
A_tile_int4  = load_global_to_shared(A^_tile)      // INT4 packed

// ===== 2. Dequantization (Shared → Registers) =====
// INT4: val = (byte >> (4*pos)) & 0xF  →  unpack
// 2:4 sparse: 从 packed weight 中根据 metadata 提取非零值
for each group of 4 in W_tile:
    (idx0, idx1) = decode_2_4_metadata(metadata[group])  // 非零位置
    val0 = unpack_int4(W_tile[group][idx0])  // 提取非零值
    val1 = unpack_int4(W_tile[group][idx1])
    // 应用到 scale: w_fp16 = val × scale_w  (反量化到 FP16)

// ===== 3. MMA (Sparse Tensor Core) =====
// PTX: mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16
//   或 mma.sp.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 (INT4 variant)
// A: 激活 (反量化后 FP16), B: 权重 (FP16 2:4 sparse)
// M=16, N=8, K=32 (per instruction), 2:4 → 有效 K=16 (50% skip)
for k_tile in K dimension:
    // Warp-level synchronized MMA
    C_reg += mma_sp_sync(A_reg[k_tile], B_sparse_reg[k_tile], metadata[k_tile])

// ===== 4. Epilogue (Write-back) =====
// FP32 accumulator → output (可 optional activation/bias)
C_global = C_reg  // 写回 global memory
```

2:4 稀疏对计算和访存的影响：
- 权重访存减少 50% (load 2 个非零值 + metadata 替代 4 个值)
- Tensor Core 计算减少 50% (跳过零值 lane)
- 理论 TOPS = (2×M×N×K×0.5) / latency（vs dense 为 2×M×N×K / latency）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NVIDIA CUTLASS 提供模板 API 实现 2:4 sparse GEMM。使用方式：(1) 用 `torch.sparse.semi_structured` 进行剪枝 → 得到 sparse W 和 metadata；(2) 用 INT4 量化 sparse W → packed INT4；(3) 调用 CUTLASS 或 TensorRT-LLM 的 INT4 sparse GEMM 执行推理。CUTLASS 3.x 支持 Mixed-Precision 和 Sparse MMA 的组合。适用场景：NVIDIA A100/H100/B100 GPU 上的 INT4 sparse LLM 推理。不适用场景：小 batch size（<16），此时 GPU 无法充分利用 Tensor Cores。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

## CUTLASS (CUDA Templates for Linear Algebra Subroutines)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUTLASS (CUDA Templates for Linear Algebra Subroutines) 是 NVIDIA 开源的 CUDA C++ 模板库，用于实现高性能 GEMM（通用矩阵乘法）和相关计算。它通过 C++ 模板将矩阵分块策略（tiling）、内存层次管理（global→shared→register）、线程块组织（warp/threadblock）和 Tensor Core 指令（mma.sync）参数化，使开发者可灵活定义和组合不同的 GEMM 配置而不损失性能。CUTLASS 是 cuBLAS/cuDNN 的底层引擎，也是 vLLM、TensorRT-LLM 等推理框架中自定义 kernel 的开发基础。CUTLASS 3.x 引入了 sm90/sm100 的 FP8/FP4/INT4 和 structured sparsity 的混合精度支持。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CUTLASS GEMM 的分层抽象（以 INT4 2:4 sparse GEMM 为例）：

```
// CUTLASS GEMM = {TileShape, WarpShape, InstructionShape, Stages, DataType}

// 1. Kernel 级 (Threadblock Tile)
using Gemm = cutlass::gemm::kernel::Gemm<
    cutlass::gemm::kernel::DefaultGemmConfiguration<
        cutlass::arch::Sm80,                    // Ampere SM
        cutlass::gemm::GemmShape<128, 128, 64>, // Threadblock Tile M×N×K
        cutlass::gemm::GemmShape<64, 64, 64>,   // Warp Tile
        cutlass::gemm::GemmShape<16, 8, 32>,    // MMA Instruction (per warp)
        int4_t,                                  // ElementA (activation)
        int4_t,                                  // ElementB (weight, 2:4 sparse)
        float,                                   // ElementC (accumulator)
        cutlass::layout::RowMajor,
        cutlass::layout::ColumnMajor,
        cutlass::arch::OpClassTensorOp           // Use Tensor Cores
    >,
    2  // Pipeline stages (double-buffering)
>;

// 2. 执行流程
// Threadblock → 分块加载 (global→shared)
// Warp → 子分块加载 (shared→register)
// MMA → Tensor Core 指令
// Epilogue → 写回

// 3. 关键配置参数
// TileShape: 每个 threadblock 处理的 M×N×K 子矩阵大小
// Stages: 软件流水线级数（>1 实现计算/访存 overlap）
// ThreadblockShape / WarpShape / InstructionShape 三层嵌套的并行粒度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/NVIDIA/cutlass (MIT License)。使用方式：(1) include CUTLASS headers；(2) 定义 Gemm kernel 类型（模板参数指定数据类型、layout、架构）；(3) 编译为 CUDA kernel；(4) 通过 `cutlass::gemm::device::Gemm` 接口调用。CUTLASS Profiler (cutlass_profiler) 可自动搜索最优 kernel 配置。OBR 使用 CUTLASS 实现 INT4 2:4 sparse GEMM 来验证实际推理加速。vLLM、TensorRT-LLM、FlashInfer 等均基于 CUTLASS 或其思想构建自定义 kernel。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
- ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals


ResQ 使用 CUTLASS 实现混合精度 INT4 + INT8 GEMM：(1) 低精度分支：CUTLASS INT4 GEMM kernel 执行 Q_L(XU_l)·Q_L(U_l^T·W)；(2) 高精度分支：CUTLASS INT8 GEMM kernel 执行 Q_H(XU_h)·Q_H(U_h^T·W)；(3) 两路结果在 INT32 累加器中求和得到最终输出。与纯 INT4 kernel 相比，ResQ 的混合精度 kernel 仅增加约 14% 延迟。实测在 NVIDIA RTX 3090 上 batch size=1 时达到 1.61×–3.03× 加速比（相比 FP16 baseline），更大模型和更短序列获得更高加速比。

---

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

## LoRunner Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRunner Kernel 是 SVDQuant (Li et al., NeurIPS 2024) 提出的用于扩散模型低比特量化的融合 CUDA kernel。其核心动机：在使用 LoRA-like 低秩分支补偿量化误差时，额外分支引入的显存访问主导了推理开销瓶颈（而非计算本身）。LoRunner Kernel 通过两个融合操作消除冗余显存访问：(1) 将低秩分支的 down projection（X → Δ = X·α）与激活量化 kernel 融合——两者共享已加载的激活张量 X；(2) 将低秩分支的 up projection（Δ·β^T → 输出）与 INT GEMM 计算 kernel 融合。这样 kernel 调用次数减半，低秩分支几乎无额外内存访问开销。在 SVDQuant 中 rank=16 时额外延迟仅 5%，在 Q-VDiT 中 rank=1 时开销更低。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LoRunner Kernel 的执行流程（以 Q-VDiT rank=1 TQE 为例）：
```
// 传统非融合执行 (3次kernel调用 + 2轮显存读写):
Kernel 1: 量化激活 = quantize(load(X), s_x, z_x)       // 读写X
Kernel 2: Δ = (M ⊙ 量化激活) @ α                         // 读X, 写Δ
Kernel 3: Y = INT_GEMM(量化激活, Q_W) + Δ @ β^T         // 读X+Δ, 写Y
// 总显存流量: 3*read(X) + write(X_q) + write(Δ) + write(Y)
// 瓶颈: 激活X被读取3次

// LoRunner 融合执行 (1次kernel调用 + 1轮显存读写):
Fused Kernel:
  __global__ void lora_quant_gemm(X, W_q, s_x, z_x, α, β, M, ...):
    // 共享显存中的 tile 加载
    __shared__ float X_tile[TILE_M][TILE_K]
    __shared__ float W_tile[TILE_K][TILE_N]

    // Step 1: 加载+量化激活 tile (down projection 融合)
    X_tile = load(X)
    X_q_tile = quantize_tile(X_tile, s_x, z_x)    // 就地量化

    // Step 2: 计算 Δ = (M ⊙ X_q) @ α (rank=1)
    for each frame i:                               // 在shared mem中完成
        Δ_local[threadIdx] += (M[i] * X_q_local) * α[threadIdx]

    // Step 3: INT GEMM + low-rank up projection
    accum = 0
    for k in range(0, K, TILE_K):
        accum += X_q_tile @ W_q_tile[k]              // 量化矩阵乘
    accum += Δ_local @ β^T                            // 低秩输出 (up proj fused)

    Y = accum
    store(Y)
// 总显存流量: 1*read(X) + write(Y) (shared mem 消去中间RD/WR)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LoRunner Kernel 的实现内置于 SVDQuant 的量化框架中：https://github.com/mit-han-lab/nunchaku。SVDQuant 论文通过 CUDA kernel 实现融合，支持可变 rank（默认 16，可降至 1）。Q-VDiT 通过调用相同的 LoRunner kernel 接口，将其应用于 Video DiT 的 TQE module (rank=1)。使用方式：(1) 在 PyTorch 模型中用 `LoraLinear` wrapper 替换标准 Linear 层；(2) 设置 rank=1 和低秩参数 α, β；(3) 推理时自动调用 fused kernel。在 Q-VDiT W4A8 Open-SORA 模型中，LoRunner 融合使得 TQE 模块在实现 2.40× 显存节省和 1.35× 推理加速的同时，引入的额外延迟可忽略（<5% vs 非 TQE 的量化推理）。

涉及论文标题：
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers

---

## Fast Hadamard Transform (快速哈达玛变换 Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast Hadamard Transform 是 Walsh-Hadamard Transform (WHT) 的高效 GPU kernel 实现。由于 WHT 矩阵元素仅为 ±1，其计算不需要浮点乘法，仅通过加法和减法即可完成。对于维度 N=2^n 的 WHT，直接矩阵-向量乘需要 O(N²) 次运算，而 Fast Hadamard Transform 利用 WHT 的递归结构（H_N = H_2 ⊗ H_{2^{n-1}}），通过类似 FFT 的蝶形运算（butterfly operations）将复杂度降至 O(N log N)。具体地，每层递归执行 N/2 对元素的加法和减法（a+b 和 a-b），共 log₂N 层，总计 N log₂N 次加减操作。Dao-AILab (2024) 提供了 CUDA 实现的 fast-hadamard-transform，通过 fused kernel 避免显式 WHT 矩阵构造，将计算融合为单次 GPU kernel launch。QWHA 论文利用该 kernel 实现适配器中的 H^{-1} X 计算：训练时用于前向传播的 ΔW=FH^{-1} 计算，推理时直接对激活 X 做 WHT 再与稀疏 F 做稀疏矩阵乘法。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fast Hadamard Transform 的递归 butterfly kernel 执行流程（GPU kernel 伪代码）：

```
// Fast Hadamard Transform Kernel (in-place, N=2^n)
// Input:  x[0..N-1]  (向量或矩阵的每一行/列)
// Output: y[0..N-1] = H_N @ x  (H_N: WHT 矩阵)

__global__ void fast_hadamard_kernel(float* x, int N, int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Butterfly stages: log2(N) iterations
    for (int step = 1; step < N; step <<= 1) {
        // step = 1, 2, 4, 8, ... N/2
        int paired = tid ^ step;  // XOR for butterfly partner
        if (tid < paired) {
            float a = x[tid];
            float b = x[paired];
            x[tid]   = a + b;    // sum
            x[paired] = a - b;   // difference
        }
        __syncthreads();
    }
    // 可选: 归一化 x /= sqrt(N)
}

// 在 QWHA 中使用场景 —— 对激活矩阵 X 的每一行做 WHT:
// X ∈ R^{d_in × (b·s)}, H^{-1} = H^T (WHT 正交)
// for each column of X (batch×seq):
//     fast_hadamard_kernel<<<grid, block>>>(X_col, d_in, 1)

// 完整的 WHA 推理前向 (融合 WHT + 稀疏 MatMul):
// Step 1: X_transformed = fast_hadamard(X)     // O(d_in log d_in) per token
// Step 2: Y_adapt = F_sparse @ X_transformed    // O(p) per token, F 仅 p 非零元
// Step 3: Y = W_Q @ X + α * Y_adapt            // 量化权重矩阵乘法
```

**与 DCT/DHT kernel 的对比**：
- WHT: 每对 (a,b) → (a+b, a-b)，2 次加减，无乘法。递归结构直接在 GPU shared memory 中完成。
- DCT/DHT: 正弦/余弦函数计算（cos, sin, cas），涉及浮点乘法，无简单 butterfly 模式 → 需显式矩阵乘法或较慢的递归 FFT kernel → 训练时间为 WHT 的 3-10x。
- 单变换 vs 双变换：QWHA 的 WHA 仅使用 1D WHT (对 d_in 维度)，而 LoCA/SSH 使用 2D DCT/DHT (同时对 d_in 和 d_out 维度)。1D WHT 训练时间 batch=4 为 6.0h，2D WHT 为 8.0h，1D DCT/DHT 为 17.4h，2D DCT/DHT 为 26.1h。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Fast Hadamard Transform 的工程实现要点（基于 Dao-AILab 的开源实现和 QWHA 论文）：
1. **无矩阵构造**：H_N 不需要显式存储为 N×N 矩阵。预计算的 H_N 仅用于需要矩阵形式的场景（如 H^{-1}R 的预计算），跨同维度层共享缓存。
2. **Fused Kernel**：将 WHT 的 log₂N 层 butterfly 融合为单个 CUDA kernel，消除中间结果的 global memory 往返。每层使用 shared memory 交换数据，仅需 __syncthreads() 同步。
3. **非 2 的幂维度**：对于 d_in 不是 2 的幂的情况，使用 H_N = H_{2^n} ⊗ H_m 分解（H_m 为已知 Hadamard 矩阵），或通过 padding 到最近的 2 的幂次。
4. **推理效率**：QWHA 中 WHA 的推理吞吐为 184.6 tok/s，仅比 LoRA (188.1 tok/s) 低 1.9%，远优于 DCA/DHA (92.4 tok/s, 下降 50.9%)。这是因为 WHT kernel 的计算开销几乎可忽略（仅加减法），而 DCT/DHT 每次变换都需要三角函数和浮点乘法。
5. **显存开销**：fast Hadamard kernel 不产生额外显存分配（in-place 操作），推理峰值显存 QWHA 52.68GB vs CLoQ 59.53GB（减少 13.0%），因稀疏适配器的 scatter ops 无额外内存。

涉及论文标题：
- QWHA: Quantization-Aware Walsh-Hadamard Adaptation for Parameter-Efficient Fine-Tuning
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，fast Hadamard CUDA kernel 用于实现在线旋转矩阵 R_3, R_3^T, R_4 的矩阵乘法（in-block online rotations），作用于 Query/Key projection（消除 KV cache outlier）和 Down projection（MLP 内）。这些在线旋转在训练和推理时均需执行，但论文指出其开销可忽略——RoSTE 训练时间 2.8h vs 无旋转 STE 2.4h（+16.7%），主要因 Hadamard kernel 的 O(d log d) 复杂度远低于主要线性层的 O(d²) 计算。RoSTE 的 fast Hadamard kernel 继承自 QuaRot/QuIP# 的开源实现，不涉及自定义 kernel 修改。
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

Quamba2 中的 Fast Hadamard Transform 用于：(1) offline Hadamard matrix fusion——将 Hadamard 矩阵 offline 融合到 input/output projection 权重（$W_{in}^H = W_{in} H_n^T$, $W_{out}^H = H_n W_{out} H_n^T$），融合后的权重与量化后的激活仍保持 compute-invariance；(2) online FWHT kernel 内联 scaling factor $s_y$，执行 $\bar{y}^H = (1/s_y) H_n \bar{y}$ 避免额外量化步骤的延迟开销。

---

## Marlin Kernel (NVFP4 Mixed-Precision Inference Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Marlin (Frantar et al., 2024) 是 IST-DASLab 开发的面向 auto-regressive LLM 推理的 mixed-precision kernel，专为低比特权重（INT4/FP4）+ FP16/BF16 激活的混合精度 GEMM 设计。QeRL 将 Marlin kernel 适配到 NVFP4×BF16 操作——利用 Marlin 的 packed 4-bit weight layout 和高效 dequant+compute 融合策略，在 H100 GPU 上实现 NVFP4 量化模型的快速 rollout 推理。关键设计：(1) weight 以 packed 4-bit 格式存储在 GPU memory；(2) kernel 内部按 block 粒度 dequantize（NVFP4: FP8 block scale S_E4M3, block size=16）并立即执行 BF16 Tensor Core GEMM；(3) 避免先全部 dequantize 再 compute 的中间数据膨胀。Marlin 的 NVFP4 支持使 QeRL 的 rollout 加速 1.2−2.0× vs BF16，7B 模型仅占 5.9GB vs BF16 15.2GB。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Marlin kernel NVFP4×BF16 GEMM 伪代码
# 输入: W_nvfp4_packed[N/2], S_FP32, S_E4M3_blocks, X_BF16
# 输出: Y_BF16 = X·Dequant(W)^T

grid = (M_tiles, N_tiles)            # 2D grid over output dims
thread_block:
    # 1. 加载 packed weights 到 shared memory
    W_packed_smem = cp_async_load(W_nvfp4_packed[tile])
    S_block_smem = cp_async_load(S_E4M3_blocks[tile])
    X_reg = load_bf16(X[tile_row])

    # 2. Dequant + MMA (fused)
    for each block (16 elements):
        w_bf16 = S_FP32 * S_E4M3[block] * unpack_4bit_to_fp16(W_packed[block])
        acc += mma_bf16(X_reg, w_bf16)  # Tensor Core warp-group MMA

    # 3. 写回结果
    Y[tile] = acc
```

对比 NF4 kernel：NF4 需 per-element lookup table（16 个 FP32 value→查表映射），无法与 GEMM 高效融合，导致 QLoRA rollout 比 BF16 还慢 0.7-0.8×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：Marlin kernel (https://github.com/IST-DASLab/marlin), QeRL 集成版 (https://github.com/NVlabs/QeRL)。使用要求：CUDA≥12.4.1, NVIDIA H100/B100/RTX 5090 (支持 NVFP4 的 GPU)。部署方式：量化权重→Marlin-optimized packed layout（与 CUTLASS/cuBLAS 的 layout 不同，需专用 repack 步骤）→将 packed weights 加载到 GPU memory→推理时 Marlin kernel 替代标准 cuBLAS GEMM。QeRL 在 GRPO rollout 阶段用 vLLM engine 调用 Marlin-accelerated NVFP4 层。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

**Sparse Marlin 变体**：SLiM 使用 Sparse Marlin（https://github.com/IST-DASLab/Sparse-Marlin）——Marlin 的 2:4 稀疏 + 4-bit 量化扩展。Sparse Marlin 将 2:4 稀疏模式编码（每 4 个连续元素保留 2 个非零）与 INT4 量化结合，利用 NVIDIA Ampere+ Sparse Tensor Core（m16n8k32 SPTC 指令）实现 FP16×INT4 稀疏-量化混合 GEMM。性能：~5.3× vs dense FP16 baseline（near-optimal for compression ratio），batch size 支持 up to 32。SLiM 在推理中将稀疏量化权重（W^C）通过 Sparse Marlin kernel 执行主矩阵乘法（Y = X · W^C），低秩适配器（LR）通过 Dense Marlin 或标准 PyTorch GEMM 执行（Y += X · L · R），两者结果相加得到最终输出。

--- in Quantized LLM Inference (量化推理中的在线 Hadamard 变换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Online Hadamard Transform 是指在 LLM 量化推理过程中，对中间激活值实时执行快速 Walsh-Hadamard 变换（FWHT）的 CUDA kernel 操作。与离线阶段将 Hadamard 矩阵融入权重不同，"在线"意味着变换在每次前向传播时对当前激活值执行——这引入了额外的计算开销（~7% for QuaRot），但使得网络结构中的非线性和不可融合操作（如 RoPE 位置编码、SiLU gating）前后的激活值也能受益于离群值消除。在线 Hadamard Transform 利用 Hadamard 矩阵仅含 ±1 元素的特性，通过 FWHT 在 O(d log d) 时间内完成，仅需加法和减法操作，无需浮点乘法。在 QuaRot 中，每层 Transformer 需要 1.5 次在线 Hadamard 变换：(1) FFN down-projection 前 1 次（处理 SiLU gating 后的激活值）；(2) attention out-projection 前 0.5 次——head Hadamard (H_{n_h}⊗I)，该变换等效于 reshape + per-head WHT。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fast Walsh-Hadamard Transform (FWHT) 的 CUDA kernel 伪代码：
```
# 输入: X ∈ R^{T×d} (d = 2^k)
# 输出: Y = X @ H_d (H_d 为归一化 Hadamard 矩阵)
# GPU grid: dim3(T, 1), block: dim3(256)

__global__ void fwht_kernel(float* X, int T, int d) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    float* x = X + token_idx * d;
    
    // In-place FWHT: log2(d) stages, butterfly pattern
    for (int step = 1; step < d; step <<= 1) {
        for (int i = tid; i < d; i += blockDim.x) {
            int partner = i ^ step;  // bitwise XOR → butterfly pair
            if (i < partner) {       // only one of each pair does the work
                float a = x[i];
                float b = x[partner];
                x[i] = a + b;
                x[partner] = a - b;
            }
        }
        __syncthreads();
    }
    // Normalize
    float scale = 1.0f / sqrtf((float)d);
    for (int i = tid; i < d; i += blockDim.x) {
        x[i] *= scale;
    }
}
```
上述 kernel 的复杂度：O(d log d) = O(4096 × 12) = 49K ops per token（对 d=4096），远小于后续矩阵乘法的 O(d × d_intermediate) ≈ 16M ops。这就是"~7% overhead"的来源。对于非 2 的幂维度 d=2^n·m：H_d = H_{2^n} ⊗ H_m，先做 m 次 FWHT_2^n，再在 2^n 个 group 内分别做 H_m 乘法（m 较小时直接用查表法）。

对于 QuaRot attention 中的 Head Hadamard (H_{n_h}⊗I)：
```
# Z ∈ R^{T × (n_h·d_h)}, n_h=32, d_h=128
# 变换 Z ← Z @ (H_{n_h} ⊗ I_{d_h})
# Kernel 策略：reshape Z to [T, n_h, d_h]，对每个 d_h 位置上的 n_h 维向量做 FWHT_nh
__global__ void head_hadamard_kernel(float* Z, int T, int nh, int dh) {
    int t = blockIdx.x;      // token
    int d = blockIdx.y;      // head_dim index
    // 加载Z[t, :, d] 到 shared memory (nh floats)
    __shared__ float s[32];  // nh ≤ 32
    s[threadIdx.x] = Z[t * nh * dh + threadIdx.x * dh + d];
    __syncthreads();
    // FWHT on nh elements (log2(nh)=5 stages)
    for (int step = 1; step < nh; step <<= 1) {
        int partner = threadIdx.x ^ step;
        float a = s[threadIdx.x], b = s[partner];
        __syncthreads();
        if (threadIdx.x < partner) { s[threadIdx.x] = a+b; s[partner] = a-b; }
        __syncthreads();
    }
    // Write back
    Z[t * nh * dh + threadIdx.x * dh + d] = s[threadIdx.x] * rsqrtf(nh);
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 使用 Dao-AILab 的 fast-hadamard-transform CUDA kernel (https://github.com/Dao-AILab/fast-hadamard-transform)；(2) 对于 FP16 精度，FWHT 的加减操作在 FP16 下精度足够（论文 Table 10 验证 FP16 vs FP32 Hadamard 困惑度差异 <0.02，零样本精度差异 <0.6%）；(3) 在线变换可以与后续量化 kernel 通过 CUDA Graphs 或 fused kernel 进一步减少 launch overhead。在 QuaRot 的 kernel benchmark（Table 14）中：INT4+FP16 Hadamard 的 W_down 层延迟为 0.403ms（4096×4096, batch=1），vs 纯 INT4 的 0.370ms，Hadamard overhead 约 8.9%。对于 KV cache decode kernel（Table 15），Hadamard overhead 更小（~7%），因为主要瓶颈是 HBM 带宽而非计算。

涉及论文标题：
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

## Kernel Fusion for Quantization + Rotation + Caching (CUDA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kernel Fusion for Quantization + Rotation + Caching 是 QuantCache 论文针对 DiT 视频生成推理提出的 CUDA kernel 优化技术。核心思想：将三个原本各自独立的运行时操作——(1) activation quantization（online uniform min-max quantize）、(2) channel-balancing rotation 变换、(3) 低精度 GEMM 矩阵乘法——融合为单次 CUDA kernel launch，配合 intermediate feature caching 逻辑，消除中间结果的 global memory round-trip 和额外的 kernel launch overhead。具体融合路径：kernel 从 global memory 加载 FP16 输入 X tile 到 shared memory → 在同一 shared memory tile 上执行 online activation quantization（计算 min/max → scale → quantize → 6-bit 或 8-bit INT）→ 执行 rotation transform（R @ X_quant，轻量矩阵乘在 shared memory 完成）→ 加载 4-bit/8-bit 量化权重 → 执行 INT8 Tensor Core GEMM → fused output dequant scaling → 写回 FP16 output 到 global memory。若无 kernel fusion，每一步需独立 kernel launch 并在 global memory 间来回传输中间数据（至少 3 次 kernel launch + 2 次 global memory round-trip per layer per timestep），在 DiT 100+ timesteps × N layers 的配置下累计 overhead 显著。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused CUDA kernel 伪代码（以 W4A6 + rotation + caching 为例）：
```cuda
// Fused Quantization + Rotation + GEMM Kernel (single launch)
__global__ void fused_quant_rotate_gemm_kernel(
    half* X,        // [seq_len, d_model] FP16 input (global mem)
    int4* W_packed, // [d_model, d_ff] 4-bit packed weights (global mem)
    float* s_W,     // per-channel weight scales (global mem)
    half* R,        // rotation matrix (global mem)
    half* Y,        // [seq_len, d_ff] FP16 output (global mem)
    half* cache,    // HLC cache buffer (global mem)
    int cache_hit   // 1=use cache, 0=compute
) {
    // 若 HLC 缓存命中，直接从 cache buffer 读取并返回
    if (cache_hit) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        Y[tid] = cache[tid];  // 复用缓存特征，跳过计算
        return;
    }

    // Step 1: 加载 X tile 到 shared memory
    __shared__ half X_shared[TILE_M][TILE_K];
    load_tile_to_smem(X, X_shared);

    // Step 2: Online activation quantization (in shared memory)
    // 计算 tile 内 min/max → scale → quantize
    float min_val = block_reduce_min(X_shared);
    float max_val = block_reduce_max(X_shared);
    float s_X = (max_val - min_val) / 63.0f;  // 6-bit: 2^6-1=63
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        int8_t X_quant = clamp(round(X_shared_flat[i] / s_X), 0, 63);
        X_shared_flat[i] = X_quant;  // in-place 量化
    }

    // Step 3: Rotation transform (fused, in shared memory)
    // X_rot = R @ X_quant (轻量矩阵乘)
    half R_tile[TILE_K][TILE_K];
    load_tile_to_smem(R, R_tile);
    half X_rot[TILE_M][TILE_K];
    tile_matmul_smem(X_shared, R_tile, X_rot);

    // Step 4: 加载 4-bit packed weights → dequant → INT8 GEMM
    int4_t W_tile[TILE_K][TILE_N];
    load_w4_tile(W_packed, W_tile);
    // INT8 Tensor Core GEMM: Y_int8 = W_deq_int8 @ X_rot_int8
    int32_t Y_acc[TILE_M][TILE_N];
    w4a8_gemm_tensorcore(W_tile, s_W, X_rot, Y_acc);

    // Step 5: Fused dequant output scaling → FP16
    half Y_out[TILE_M][TILE_N];
    #pragma unroll
    for (int i = 0; i < TILE_M * TILE_N; i++) {
        Y_out_flat[i] = (half)(Y_acc_flat[i] * s_X * s_W_block);
    }

    // Step 6: 写回 output + 更新 HLC cache buffer
    store_tile_to_gmem(Y_out, Y);
    if (D_t_l < delta_1)  // HLC 判定可缓存
        store_tile_to_gmem(Y_out, cache);  // 写入 cache buffer
}
```
Kernel fusion 收益：kernel launch overhead 从 3× 降至 1×；消除 2 次 global memory round-trip（quantized X + rotated X）；shared memory 内完成 quantize+rotate 避免 HBM 访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QuantCache 的 fused kernel 基于 CUDA C++ 实现（CUDA 12.1），在 NVIDIA A800-80GB (Ampere) GPU 上运行。实现要点：(1) 使用 CUDA Tensor Core 的 INT8 矩阵乘法指令（如 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`）；(2) shared memory 作为 quantize+rotate+GEMM 的中间工作区——tile 大小需平衡 shared memory 容量限制（A800 48KB per SM）和 Tensor Core tile 对齐要求；(3) HLC cache buffer 分配在 GPU global memory，大小 = num_layers × feature_size × FP16；(4) scale factor absorption 受 QServe（Lin et al., MLSys 2025）启发，将 channel-balancing scaling offline 融合到前层权重避免额外计算；(5) SRAP 层剪枝在 kernel 调用侧（host-side）判断——cosine similarity 在 CPU/轻量 GPU kernel 中计算，若 S > τ_high 则完全跳过该 kernel launch。开源代码：https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

## INT8 Tensor Core MMA for Attention (INT8 Tensor Core 注意力矩阵乘加)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
INT8 Tensor Core MMA for Attention 是指在 Attention kernel 中利用 NVIDIA GPU Tensor Core 的 INT8 矩阵乘加指令（mma.u8.u8.s32）来加速 QK^⊤ Matmul。SageAttention 的选择理由：(1) INT8 throughput 在 consumer GPU (RTX4090) 上理论为 660 TOPS——是 FP16 (330 TFLOPS) 的 2×、FP8 (330 TFLOPS) 的 2×；(2) 实测 340 TOPS at headdim=64，达到理论峰值的 52%（FlashAttention2 仅 165 TOPS，50% FP16 峰值）；(3) SageAttention 在同一 kernel 内交替使用 INT8 (for QK^⊤) 和 FP16 (for PV) MMA 指令。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SAGEAttn-B kernel (single Triton kernel launch)
# Q̂, K̂, V tiles 从 HBM 加载到 SRAM

# INT8 MMA for QK^⊤:
S_int32 = tl.dot(Q̂_i_INT8, K̂_j_INT8^T, input_precision='int8')  
# ↑ Triton → PTX: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
# u8 inputs × 2 = 16-bit intermediates, accumulate to s32

# Dequantization (in FP16):
S_ij = S_int32.to(tl.float16) * δ_Q[i] * δ_K[j]
# ↑ per-block scale broadcast, light element-wise op

# Online Softmax + FP16 MMA for PV:
P̃_ij = exp(S_ij - m_new)  # FP16 exp
O += tl.dot(P̃_ij.to(tl.float16), V_j.to(tl.float16), out_dtype=tl.float16)
# ↑ FP16+FP16 accum MMA, f16.f16.f16
```
Kernel 配置（Table 12）: Q block b_q=128, KV block b_kv=64; Num Warps=4 (headdim=64) or 8 (headdim=128); Num Stages=3-5。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Triton 通过 `tl.dot()` 自动映射到 Tensor Core 指令。实现关键：(1) 输入张量必须是 int8 dtype (`tl.int8`)，值域 [-127, 127]；(2) Triton compiler 自动处理 Tensor Core tile 对齐（M=16, N=8, K=32 for INT8）；(3) 混合精度 kernel——在同一 Triton program 中先后调用 INT8 MMA 和 FP16 MMA——Triton 自动插入必要的 dtype conversion 指令；(4) 性能瓶颈分析：当 sequence length 较小（<512）时 attention 受 kernel launch overhead 主导，INT8 加速效果有限；当 sequence length 较大（>2048）时 compute-bound，INT8 MMA 加速效果显著。开源: https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## FlashAttention-2 Tiling Strategy (FlashAttention-2 Tiling策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashAttention-2 Tiling Strategy 是 FlashAttention-2 (Dao, 2023) 提出的将 Attention 计算分解为小块（tiles）以消除 N×N 中间矩阵 HBM I/O 的算法-系统协同设计。核心思想：将 Q 沿 token 维分为 b_q 大小的 tiles {Q_i}，K,V 分为 b_kv 大小的 tiles {K_j},{V_j}，使用 online softmax 逐步累加 O_i，使得 N×N 的 S 和 P 矩阵永远不需要整体写入 HBM。FlashAttention-2 改进点：减少 non-matmul FLOPs、优化 parallelism（outer loop on Q 并行于 SMs、inner loop on KV 串行）、优化 warp partition。SageAttention 在 FlashAttention-2 tiling 基础上叠加 INT8 量化——Q,K tile 在加载到 SRAM 后先量化为 INT8，再执行 Tensor Core MMA。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FlashAttention-2 Tiling (SageAttention adaptation)
# Grid: T_m = N/b_q SM blocks, each handles one Q tile
# Thread block i (runs on one SM):
Load Q̂_i_INT8 [b_q×d] + δ_Q[i] from HBM to SRAM     # outer tile
for j in 1..T_n:                                      # inner loop
    Load K̂_j_INT8 [b_kv×d] + V_j_FP16 [b_kv×d] + δ_K[j] to SRAM
    S_ij = INT8_MMA(Q̂_i, K̂_j^T) * δ_Q[i] * δ_K[j]    # [b_q×b_kv], SRAM resident
    O_i = online_softmax_update(O_i, S_ij, V_j)       # in SRAM/registers
Write O_i_FP16 [b_q×d] to HBM
```
SageAttention 的 block sizes: b_q=128, b_kv=64（比 FlashAttention-2 默认的 b_q=128 和 b_kv=64 保持一致）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashAttention-2 是当前 GPU attention 计算的事实标准。CUDA 实现（原版）：直接编写 CUDA C++ kernel，管理 shared memory allocation（Q_tile + K_tile: 2×b_q×d×2bytes + 2×b_kv×d×2bytes）。Triton 实现（SageAttention）：`tl.dot()` 自动管理 shared memory。与标准 FlashAttention-2 的差异：(1) SageAttention 在 K 加载后 fuse smooth K（减去 mean）和 INT8 量化；(2) QK MMA 从 FP16→INT8；(3) PV MMA 从 FP16+FP32 accum→FP16+FP16 accum。tiling pattern（outer Q loop + inner KV loop）保持不变。开源: FlashAttention-2 https://github.com/Dao-AILab/flash-attention, SageAttention https://github.com/thu-ml/SageAttention。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

## ROPE + Quantization Kernel Fusion (ROPE与量化Kernel融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ROPE + Quantization Kernel Fusion 是 SageAttention 将 Rotary Position Embedding (RoPE) 操作与 Q/K 的 INT8 量化融合在同一个 GPU kernel 中完成的技术。目的：消除量化引入的额外 HBM I/O overhead。传统非融合方案：ROPE kernel: read Q,K from HBM → apply rotary → write Q_rope,K_rope to HBM；Quant kernel: read Q_rope,K_rope from HBM → quantize → write Q̂,K̂,δ_Q,δ_K to HBM。两次 HBM round-trip。融合方案：ROPE kernel 计算完 rotary 结果后，在 shared memory 中直接进行量化（on-chip），然后将 INT8 Q̂,K̂ 和 FP16 scales 写入 HBM，节省一次 round-trip。此外，SageAttention 还将 1/√d 系数融合到 Q 量化中（在 ROPE 后将 Q 乘以 1/√d，再量化），避免在 attention kernel 中额外做除法。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 非融合方案 (2 kernels, 2 HBM round-trips):
# Kernel 1 (ROPE):
Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)
write Q_rope, K_rope to HBM
# Kernel 2 (Quantize):
read Q_rope, K_rope from HBM
Q̂, δ_Q = per_block_quantize_int8(Q_rope * (1/sqrt(d)))
K̂, δ_K = per_block_quantize_int8(K_rope)
write Q̂, K̂, δ_Q, δ_K to HBM

# 融合方案 (1 kernel, 1 HBM round-trip):
# Kernel (Fused ROPE + Quant):
read Q, K from HBM
Q_rope, K_rope = apply_rotary_pos_emb(Q, K, cos, sin)  # register/SRAM
Q_scaled = Q_rope * (1/sqrt(d))                          # on-chip scale
Q̂, δ_Q = per_block_quantize_int8(Q_scaled)               # on-chip quant
K̂, δ_K = per_block_quantize_int8(K_rope)                 # on-chip quant
write Q̂, K̂, δ_Q, δ_K to HBM                              # single write
```
该融合减少了量化的 I/O overhead，使 quantize step 的额外 overhead 被 ROPE 的计算开销所 overlap。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Triton 实现：在 ROPE Triton kernel 的 epilogue 中插入量化逻辑。Triton 的 single-program 模型使得在同一 kernel 内混合 memory-bound (ROPE) 和 compute-bound (quantize) 操作变得自然。具体步骤：(1) ROPE 将 Q,K 从 HBM 加载到 SRAM；(2) 应用 rotary embedding（element-wise sin/cos 乘加）；(3) 在写入 HBM 前，在 SRAM 中计算 `max(|x_tile|)` 得到 scale，执行 `x̂ = round(clamp(x/scale, -127, 127)).to(tl.int8)`；(4) 将 INT8 数据和 FP16 scales 写入 HBM。该融合对 end-to-end 延迟有实质贡献——消除了量化 kernel 的单独 launch overhead 和一次 full tensor HBM round-trip。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

---

## Group-wise Mixed-Precision Dequantization Kernel (AutoGPTQ Extension)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
该 kernel 是 SliM-LLM 基于 AutoGPTQ 扩展的 CUDA 推理 kernel，支持 group 级别混合精度（1/2/3-bit）权重的 GPU dequantization 和矩阵乘法。核心设计：(1) Weight Packing——量化后的权重按 group 分别 pack 为整数（每个 group 内 128 个元素以相同精度 pack，利用 group_size=128 是任意 2 的幂的倍数这一特性，无需 padding）；(2) 额外 bit-widths array——每个 group 用 2-bit 编码精度（00=未使用, 01=1-bit, 10=2-bit, 11=3-bit），聚合成 32-bit 整数数组；(3) 逐 group 解包计算——GPU 上每个 thread 处理一列连续 pack 数据的 dequantization，与 block 内共享的 input activation 做向量点积，结果累加到输出矩阵对应位置。因为精度在 group 边界对齐（而非 element-wise），warp 内 32 threads 的 code path 和数据访问逻辑保持一致。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 每个Logical Block处理一段连续channel区域
__global__ void mixed_precision_dequant_matmul(
    uint32_t* w_packed,    // 混合精度packed权重
    uint32_t* z_packed,    // packed zeros
    half* scales,          // FP16 scales
    uint32_t* bit_widths,  // 每group的2-bit精度编码
    half* input,           // input activation [t, m]
    half* output           // output [t, n]
) {
    // 加载共享的input activation片段
    __shared__ half input_shared[BLOCK_SIZE];
    // 共享activation加载到shared memory
    load_input_to_shared(input, input_shared);

    // 逐group处理
    int group_offset = 0;  // 追踪packed数组中的累积偏移
    for (int g = block_start_group; g < block_end_group; g++) {
        // 读取当前group的bit-width
        int bw_idx = g / 16;  // 每32-bit存16个group的精度(每组2-bit)
        int bw_shift = (g % 16) * 2;
        int bw = (bit_widths[bw_idx] >> bw_shift) & 0x3;  // 提取2-bit

        // 根据bw计算packed元素数
        int elems_per_int = 32 / bw;  // 1-bit:32, 2-bit:16, 3-bit:10(余2)
        int idx_in_int = thread_id % elems_per_int;

        // 解包该thread对应的权重值
        uint32_t packed_val = w_packed[group_offset + thread_id / elems_per_int];
        int w_int = (packed_val >> (idx_in_int * bw)) & ((1 << bw) - 1);

        // 反量化
        half w_deq = (half)(w_int - zero) * scale;

        // 向量点积累加
        for (int t = 0; t < num_tokens; t++) {
            output[t * n + out_col] += w_deq * input_shared[t];
        }

        group_offset += (128 + elems_per_int - 1) / elems_per_int;  // 累积偏移
    }
}
```
与统一精度 kernel 的关键差异：bit-width 需要逐 group 读取和解析（额外 2-bit/group 的 array lookup）；累积偏移计算确保跨 group 的正确 start index；1-bit group 需要 sign+α 反量化（而非标准 INT dequantization），增加了分支。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 kernel 基于 AutoGPTQ 的 CUDA warp 机制实现。每个 warp 内 32 threads 处理一个 group（128 元素，组大小正好是 warp size 的倍数），使 threads 在相同 code path 上执行。实现的主要挑战：(a) 1-bit 权重使用 sign + α 格式反量化（ŵ = α · sign(w_fp)），不走标准 INT 量化路径，需要额外分支处理；(b) 3-bit 权重 128 个元素 pack 为 10 个 32-bit 整数的子集（10×32=320 bits > 128×3=384 bits → 需要 12 个 32-bit int），存在跨整数边界对齐问题。部署验证：LLaMA-7B 2-bit SliM-LLM inference 61.2 token/s vs GPTQ 2-bit 83.9 token/s（~27% slowdown），换取 90% perplexity 提升（PPL 14.58 vs 152.31）。开源代码：https://github.com/Aaronhuang-778/SliM-LLM。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

## CSR Sparse Matrix Multiplication for LLM Inference（面向 LLM 推理的 CSR 稀疏矩阵乘法 CUDA Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSR (Compressed Sparse Row) 稀疏矩阵乘法用于 LLM 推理中的 hybrid sparse-dense weight format——当约1%的权重以非结构化稀疏格式保留为高精度（FP16），其余权重以低位量化存储时，需要专门的 GPU kernel 高效执行 sparse + dense 两部分矩阵乘法的叠加。SpQR 论文为 outlier weights 设计了定制 CUDA kernel：(1) outlier 以 CSR 格式存储（按 row-first/column-second 排序，每个 outlier: FP16 value + FP16 col_index，每行一个 cumulative row pointer）；(2) kernel 通过 tile-based 划分实现 load balancing——将权重矩阵划分为等大小 blocks，每个 thread block 加载其 tile 覆盖的 outlier slice 到 shared memory (SRAM)；(3) 每个 GPU core 判断其 tile 范围的 rows 中哪些含有 outlier，仅对有效行加载对应 col_index 和 value；(4) 执行 sparse dot product 累加到 dense dequantized 结果上。因 outlier 的 row-wise pattern，column index/value 的内存访问趋于连续。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SpQR CSR Sparse MatMul Kernel 伪代码（在 A100 上 batch_size=1 token-by-token 生成）：

```cuda
// Input:  activation x (FP16, d_in)
//         CSR: row_ptr[N+1], col_idx[num_outliers], values[num_outliers]
//         dense_quant: Q (packed INT3/4), scales_s (FP16), scales_z (FP16)
// Output: y = W_dense × x + W_sparse × x

// === Dense dequant + matmul (先执行) ===
for each thread block (tile of output rows):
    for each block (beta1 x beta2 weights):
        // 1. Load bilevel statistics to SRAM
        // 2. Second-level dequant: scales/zeros (3-bit→FP16)
        // 3. First-level dequant: weights (3-bit→FP16)
        // 4. Dot product with activation segment
        partial_dense[tile] += dot(weights_fp16, x_segment)

// === Sparse CSR matmul (后执行) ===
// Tile partitioning
tiles = divide_output_rows_into_tiles(W_dense.rows, TILE_SIZE)

for each thread_block b (maps to tile t):
    // Step 1: Identify outlier range for this tile
    row_start = tiles[t].start_row
    row_end = tiles[t].end_row
    outlier_start = row_ptr[row_start]
    outlier_end = row_ptr[row_end]
    num_tile_outliers = outlier_end - outlier_start

    // Step 2: Load outlier slice to shared memory
    __shared__ uint32_t smem_col_idx[MAX_TILE_OUTLIERS]
    __shared__ half   smem_values[MAX_TILE_OUTLIERS]

    for i = threadIdx.x; i < num_tile_outliers; i += blockDim.x:
        smem_col_idx[i] = col_idx[outlier_start + i]
        smem_values[i] = values[outlier_start + i]
    __syncthreads()

    // Step 3: Per-row sparse dot product
    for row = row_start + threadIdx.x; row < row_end; row += blockDim.x:
        o_start = row_ptr[row] - outlier_start
        o_end = row_ptr[row+1] - outlier_start
        acc = 0.0f
        for k = o_start; k < o_end; k++:
            acc += smem_values[k] * x[smem_col_idx[k]]
        partial_sparse[row] = acc
    __syncthreads()

// === Merge ===
y[row] = partial_dense[row] + partial_sparse[row]
```

Load balancing 关键：步骤1-3通过 tile 划分确保每个 thread block 处理的outlier数量大致均匀，步骤4因row-wise outlier pattern获得连续内存访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SpQR 开源：https://github.com/Vahe1994/SpQR。CUDA kernel 以 C++/PTX 实现。核心实现要点：(1) 利用 PyTorch CUDA Extension 编写自定义 kernel（而非 cuSPARSE 通用接口），因为通用稀疏库的 format conversion 和间接内存访问开销过大；(2) 在 token-by-token 生成（batch_size=1, memory-bound）场景下，因压缩率 >3.4x，DRAM 读取量大幅减少，即使额外 sparse compute 开销存在，wall-clock time 仍比 FP16 推理快 20-30%。相比之下，PyTorch 默认的 cuSPARSE sparse matmul 比 FP16 推理更慢（因稀疏矩阵乘法在低 batch size 下 overhead 较大）。该 kernel 与 dense quantized matmul 串联使用，未融合（论文将其列为 future work）。

涉及论文标题：
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

## LUT-Based Dequantization Kernel (Non-Uniform Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-Based Dequantization Kernel 是用于非均匀量化权重推理的 CUDA kernel。权重以 b-bit indices（3/4-bit）存储，每个 output channel 对应一个 LUT（Look-Up Table，包含 2^b 个 FP16 centroid 值）。Kernel 在 GPU 上加载 packed bit indices → 逐 piece 查表还原为 FP16 权重 → 与 FP16 activation vector 进行矩阵-向量乘法（GEMV）。关键设计：(1) piece-by-piece dequantization 以减少寄存器压力和最大化内存带宽利用；(2) 所有算术在 FP16 执行；(3) LUT 存储在 GPU shared memory 或寄存器中以减少查表延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// CUDA kernel: LUT-based dequant + GEMV
// grid: num_output_channels / BLOCK_SIZE
// each thread handles one output row

__global__ void lut_dequant_matvec_kernel(
    const uint32_t* packed_indices,  // [out_c × chunks_per_row]
    const half* LUTs,                 // [out_c × k], k=2^bit FP16 centroids
    const half* activation,           // [in_features] FP16
    half* output                      // [out_features]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    half* lut_row = LUTs + row * k;  // 当前 channel 的 LUT
    half acc = 0.0;

    for (int chunk = 0; chunk < num_chunks; chunk++) {
        uint32_t packed = packed_indices[row * num_chunks + chunk];
        // 逐元素提取 b-bit index, LUT查表, FP16乘累加
        for (int j = 0; j < indices_per_chunk; j++) {
            int idx = (packed >> (j * BIT_WIDTH)) & ((1 << BIT_WIDTH) - 1);
            half w_deq = lut_row[idx];               // LUT查表
            acc += w_deq * activation[global_col++]; // FP16 FMA
        }
    }
    output[row] = acc;
}
```

延迟分析（A6000, LLaMA-7B, 128 tokens, 3-bit）：
- FP16 baseline (no quant): 3.2s
- Uniform quant (GPTQ, no group): 1.4s
- LUT non-uniform (SqueezeLLM): 1.5s (+7% vs uniform, 2.1x vs FP16)
→ LUT overhead 极小，因为推理是 memory-bound（memory bandwidth 掩盖了 LUT 查表计算）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源实现：https://github.com/SqueezeAILab/SqueezeLLM (CUDA kernels)。实现要点：(1) BIT_WIDTH=3 时用 custom bit extraction（CUDA 无原生 3-bit 类型），将 3-bit indices 紧密 pack 进 uint32；(2) LUT 大小：对 4096 output channels × 8 centroids × 2 bytes = 64KB per layer——可放入 L1 cache/shared memory；(3) 相比 uniform dequant（只需 scale × int + zero），LUT 多了一次 memory read（LUT lookup），但这在 memory-bound 场景下几乎不增加 wall-clock time；(4) 与 uniform quant kernel 的关键区别：uniform kernel 按 group 读取 scale/zero point → linear dequant，非均匀 kernel 按 element 或 sub-chunk 读取 index → LUT-based dequant。非均匀量化 kernel 的通用性：可用于任何基于 codebook/centroid 的量化方案（如 GPTVQ 的 1D VQ、NF4 等）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization

## Balanced CSR Sparse Matrix-Vector Multiplication Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Balanced CSR SpMV kernel 是处理高度不均衡稀疏模式的矩阵-向量乘法 CUDA kernel。标准 CSR kernel 将每行分配给一个线程处理，但当各行的 nonzero 数量严重不均衡时（如某些通道含远超平均的 outliers），会导致严重的线程间负载不均衡（部分线程空闲等待，部分线程执行大量计算）。Balanced kernel 改为按固定 nonzeros/thread（如 10 nz/thread）分配工作——一行可由多个线程合作处理，每线程处理固定数量的连续 nonzero 元素。代价是需要 atomicAdd 合并同一行的多线程部分结果，但在高度不均衡的稀疏模式下总体收益远大于同步开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Balanced CSR SpMV kernel (per-thread固定nonzeros, 非per-row)
// 处理 Dense-and-Sparse decomposition 的 sparse matrix S

__global__ void balanced_csr_matvec_kernel(
    const int* row_ptrs,        // [out_rows + 1]
    const short* col_indices,   // [total_nnz]
    const half* values,         // [total_nnz]
    const half* activation,     // [in_features]
    half* output                // [out_rows], 需初始化为0
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nz_per_thread = 10;
    int nz_start = tid * nz_per_thread;
    int nz_end = min(nz_start + nz_per_thread, total_nnz);

    half local_acc = 0.0;
    int current_row = -1;
    half row_acc = 0.0;

    for (int nz = nz_start; nz < nz_end; nz++) {
        int col = col_indices[nz];
        half val = values[nz];

        // 确定该 nonzero 所属的行
        int row = binary_search_row(row_ptrs, nz);
        if (row != current_row) {
            if (current_row >= 0)
                atomicAdd(&output[current_row], row_acc);
            current_row = row;
            row_acc = 0.0;
        }
        row_acc += val * activation[col];
    }
    if (current_row >= 0)
        atomicAdd(&output[current_row], row_acc);
}
```

性能对比（A6000, LLaMA-7B 3-bit + 0.45% sparsity, 128 tokens）：
| Kernel Type | Latency (s) |
|------------|-------------|
| Dense-only (0% sparse) | 1.5 |
| Standard CSR (0.45%) | 3.9 (+160%) |
| Balanced CSR (0.45%) | 1.7 (+13%) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源实现：https://github.com/SqueezeAILab/SqueezeLLM。基于 Flegar & Quintana-Ortí (Euro-Par 2017) 的 balanced CSR 方法。关键参数：10 nonzeros/thread 是论文实验的最佳设置。适用场景：(1) 稀疏矩阵中 per-row nonzero 分布高度 skewed（少数行含大量 nonzero，多数行几乎为空），常见于 Dense-and-Sparse decomposition 中 outliers 集中在特定 attention head/output channel；(2) batch_size=1 的 memory-bound 推理场景。与 cuSPARSE 通用 SpMV 的对比：(1) cuSPARSE 在低 batch size + 低 sparsity 下 overhead 大（format conversion, 间接内存访问），可能比 FP16 dense 推理更慢；(2) Balanced kernel 专为该场景设计，与 dense LUT kernel 融合在单次 launch 中。局限性：atomicAdd 在极多线程竞争同一行时可能成为瓶颈（虽然通常不会，因为 skewed 分布意味着大部分行只有少量 nonzero）。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization


## Fused RoPE Kernel with Symmetric Sorting (融合RoPE Kernel与对称排序)

术语是什么？
Fused RoPE Kernel with Symmetric Sorting 是 UniQL 为支持结构化剪枝后的 LLM 推理而设计的融合 CUDA kernel。其核心挑战：结构化权重排序破坏了 Rotary Position Embedding (RoPE) 的原始位置嵌入索引顺序——因为 Q 和 K 投影矩阵的列被排序矩阵 $\mathbf{S}_{qk}$ 重排，使得第 j 列的 RoPE 嵌入不再对应原始的第 j 个维度分量。UniQL 通过对称排序 + 融合索引 gather 在单个 kernel 中解决此问题。

**对称排序策略**：RoPE 按维度对 $(2d, 2d+1)$ 应用旋转：$\operatorname{RoPE}(\mathbf{x}; \theta) = [\cos\theta_d \cdot \mathbf{x}_{2d} - \sin\theta_d \cdot \mathbf{x}_{2d+1}, \sin\theta_d \cdot \mathbf{x}_{2d} + \cos\theta_d \cdot \mathbf{x}_{2d+1}]_{(2d,2d+1)}$。为在排序后保持 RoPE 的正确语义，UniQL 将 norm score 向量 $s \in \mathbb{R}^{D_{hd}}$ 对半分：$[s_1, s_2] = s$，然后对 $s_1 + s_2$ 排序（而非独立排序 $s_1$ 和 $s_2$），得到对称索引向量 $idx_{sym} = [\operatorname{argsort}(s_1 + s_2), D_{hd}/2 + \operatorname{argsort}(s_1 + s_2)]$。这保证每个 RoPE 维度对的相对顺序不变——对于原维度对的 $(d, d + D_{hd}/2)$，排序后变为对应的新位置对 $(d', d' + D_{hd}/2)$。

从kernel调度角度拆解：
```
# Kernel 输入
# - X_q/X_k: [T, D'_hd] 或 [T, D_hd], 已排序的 Q/K 激活
# - idx_sym: [D'_hd], 对称排序索引 (前半 = 后半 + D_hd/2)
# - cos_table, sin_table: [T, D_hd], 原始 RoPE 嵌入表

# 传统两阶段实现 (无融合):
# Stage 1: gather cos/sin 索引对应的值
cos_k = cos_table[:, idx_sym]                   # [T, D_hd], global mem read
sin_k = sin_table[:, idx_sym]                   # [T, D_hd], global mem read
# Stage 2: 应用 RoPE 旋转
for d in range(0, D_hd, 2):
    x0, x1 = X_k[:, d], X_k[:, d+1]
    X_k_rope[:, d]   = cos_k[:, d] * x0 - sin_k[:, d] * x1
    X_k_rope[:, d+1] = cos_k[:, d+1] * x1 + sin_k[:, d+1] * x0

# UniQL 融合 Kernel (单个 CUDA kernel):
__global__ void fused_rope_kernel(
    half* X, half* out,
    const half* cos, const half* sin,
    const int* idx_sym,
    int T, int D_hd
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_D = D_hd / 2;
    
    // 每个线程处理一对 RoPE 维度
    int d = tid * 2;
    if (d >= D_hd) return;
    
    int idx0 = idx_sym[d];         // 索引对的前半
    int idx1 = idx_sym[d + 1];     // 索引对的后半 (自动 = idx0 + half_D)
    
    for (int t = 0; t < T; t++) {
        half cos0 = cos[t * D_hd + idx0];
        half sin0 = sin[t * D_hd + idx0];
        half cos1 = cos[t * D_hd + idx1];
        half sin1 = sin[t * D_hd + idx1];
        
        half x0 = X[t * D_hd + d];
        half x1 = X[t * D_hd + d + 1];
        
        // 直接使用 gather 后的 cos/sin 计算 RoPE
        out[t * D_hd + d]     = cos0 * x0 - sin0 * x1;
        out[t * D_hd + d + 1] = cos1 * x1 + sin1 * x0;
    }
}
```

术语一般如何实现？如何使用？
Kernel 基础实现改编自 Liger-Kernel (Hsu et al., 2025) 的 RoPE kernel 和 Marlin (Frantar et al., 2024) 的 4-bit kernel。关键优化：①对称排序将索引存储减半（仅需前半 $D_{hd}/2$ 个索引，后半由 $+D_{hd}/2$ 隐式推导），节省寄存器/共享内存；②融合 gather + slice + RoPE 旋转在单个 kernel 中，避免多次 global memory 往返。在 A6000 上 Profile 结果（Table 9）：4-bit Llama-3.1-8B at 0% 剪枝率下 TPOT 从 9.9ms（无融合）降至 9.0ms（融合），10% latency reduction；at 25% 剪枝率从 8.6ms 降至 7.7ms（1.1× speedup）。Qwen-2.5-7B 同样显示 9.1ms → 8.3ms 和 7.9ms → 7.1ms 的改善。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs

