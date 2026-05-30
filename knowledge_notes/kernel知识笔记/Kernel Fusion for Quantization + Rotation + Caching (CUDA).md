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
