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
