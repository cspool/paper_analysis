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
