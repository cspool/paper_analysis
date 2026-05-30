## SpGEMV (Sparse GEMV) with Quantized Key Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SpGEMV (Sparse Generalized Matrix-Vector multiplication) with Quantized Key Cache 是 Twilight 提出的高效 attention weight 估计 kernel。核心设计：对 K cache 做 per-head asymmetric INT4 quantization，在 GPU 上执行 q_fp16 @ K_int4 的稀疏矩阵向量乘法来估计 attention weights，同时将 memory access 降至 FP16 K cache 的 1/4。这是 top-p sparse attention 的关键性能 enabling 技术——top-p 比 top-k 需要更高的数值精度（不仅序数正确，还需数值准确性），因此不能使用极低精度（1-2 bit），但 FP16 又浪费带宽。实验证明 4-bit 是最优 sweet spot。

从kernel调度角度拆解术语，给出具体例子。
```
// INT4 K Cache SpGEMV kernel (基于FlashInfer decode attention kernel修改)
Input: q ∈ R^{BS×H×d}, K_int4 ∈ R^{N×d/2} (paged, per-head dynamic quantized)
Output: W_approx ∈ R^{BS×H×N}  (estimated attention weights)

Per thread block:
  // 2-stage software pipeline
  for k_iter in 0..d/Kt:
    // Stage 1: async load + dequantize current tile
    cp.async: GMEM[K_int4[k_iter]] → SMEM[buf_ping]   // load INT4 K tile
    cp.async.commit_group
    cp.async.wait_group
    
    // Dequantize in shared memory:
    // K_fp16 = (K_int4_unpacked - zero) * scale
    // Use per-head dynamic scale/zero stored in paged layout
    // INT4→FP16 conversion via PTX asm (FasterTransformer-style)
    K_tile_fp16 = dequantize_int4_to_fp16(SMEM[buf_ping])
    
    // Stage 2: dot product (overlapped with next tile async load)
    for i in range(BS×H):
      W_approx[i, :] += dot(q[i, k_iter:k_iter+Kt], K_tile_fp16)
    
    // swap ping/pong buffers
    
  return W_approx
```

Bit-packing: 两个 INT4 元素打包为 uint8_t（interleaved packing），地址计算 remap 到 4-bit granularity（halving effective byte offset）。Dequantization 前先加 offset +128 转 unsigned 再 pack。

术语一般如何实现？如何使用？
基于 FlashInfer 的 attention decoding kernel 修改，自研 CUDA/Triton kernel。INT4 K cache 使用 paged layout 与 FP16 KV cache 对齐。Per-head 动态量化参数（FP16 scale + zero）同样使用 paged 布局。额外内存开销：1/8 FP16 KV cache。该 kernel 也可复用于其他需要估计 attention weights 的 sparse attention 方法。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning
