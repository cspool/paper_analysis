## Online Outlier Detection and Quantization Fused Kernel（在线异常值检测量化融合kernel）

术语是什么？通过联网搜索让回答具体和精准。

Online Outlier Detection and Quantization Fused Kernel是RoMeo中使用Triton实现的融合kernel，将per-token outlier detection（row-max reduction + top-k selection）、mixed precision quantization（INT4 + INT8）和INT4 data packing合并为单一GPU kernel，消除多次kernel launch和中间global memory round-trip的开销。由于token-wise outlier来源于input text的语言特征而非静态模型结构，必须在serving时在线检测。

从kernel调度角度拆解术语：

RoMeo的fused Triton kernel伪代码：

```
// Fused Triton Kernel: Outlier Detection + Quantization + Packing
// Grid: (M // BLOCK_M, 1), each block processes BLOCK_M token rows

Kernel FusedOutlierDetectQuantize:
  Input:  X_rot [M, K] (FP16, after Hadamard rotation)
  Output: X_Q_int4 [M, K//2] (packed INT4, 2 elements per byte)
          X_Q_int8 [k_a, K] (outlier buffer, INT8)
          scales_int4 [M] (FP16 per-token scaling factors)
          scales_int8 [k_a] (FP16 per-token scaling factors)
          outlier_mask [M] (bool)

  pid = program_id(0)
  row_start = pid * BLOCK_M
  row_end = min(row_start + BLOCK_M, M)

  // === Phase 1: Per-token row-max reduction ===
  // 每个program处理BLOCK_M个token rows
  shared_max [BLOCK_M]  // shared memory for row-wise max
  for i in row_start..row_end:
    local_max = 0.0
    for k_block in 0..K/BLOCK_K:
      tile = load(X_rot[i, k_block*BLOCK_K : (k_block+1)*BLOCK_K])
      local_max = max(local_max, max(|tile|))
    shared_max[i - row_start] = local_max

  // === Phase 2: Top-k outlier selection (within block) ===
  // 使用shared memory做block-local topk
  sorted_idx = argsort(shared_max, descending=True)
  for j in 0..min(k_a_per_block, BLOCK_M):
    outlier_mask[sorted_idx[j]] = True

  // === Phase 3: Mixed precision quantization ===
  for i in row_start..row_end:
    row = X_rot[i, :]
    if outlier_mask[i]:
      // INT8 quantization: range [-127, 127]
      scale = max(|row|) / 127.0
      X_Q_int8[outlier_idx] = round(row / scale)
      scales_int8[outlier_idx] = scale
      outlier_idx += 1
    // INT4 quantization: range [-7, 7] (always quantize for W4A4)
    scale = max(|row|) / 7.0
    X_Q_int4_packed[i] = pack_int4(round(row / scale))
    scales_int4[i] = scale

  // === Phase 4: INT4 data packing ===
  // 每2个INT4元素pack为1 byte: [elem0 | elem1<<4]
  // 压缩比 2:1, K个INT4元素→K/2 bytes
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现方式：Triton编写，融合了row-max reduction（parallel reduction over K dim）、topk selection、round-to-nearest quantization、scaling factor计算和INT4 packing。Triton的block-level programming model适合这种fused算子。
- 开销：RoMeo实测所有在线overhead（Hadamard + outlier detection + quantization + post-mul overwrite）共占layer latency约12%（batch=64时），其中outlier detection+quantization占约4%。
- 与offline outlier detection对比：channel-wise方法（MixQ）可用offline calibration static确定outlier→无在线detection开销。Token-wise方法（RoMeo）必须在线→fused kernel是minimize overhead的关键。
- INT4 packing格式：NVIDIA Tensor Core要求INT4以特定packed layout存储（每byte两个元素，even element在低4 bit、odd element在高4 bit），Triton的`tl.store`配合合适的block pointer可直接生成正确packing。
- 开源：https://github.com/thu-pacman/RoMeo

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

