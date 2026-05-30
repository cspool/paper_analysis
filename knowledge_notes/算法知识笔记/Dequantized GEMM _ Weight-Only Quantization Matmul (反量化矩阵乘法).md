## Dequantized GEMM / Weight-Only Quantization Matmul (反量化矩阵乘法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dequantized GEMM (反量化矩阵乘法) 是 Weight-Only Quantization 场景下的核心计算操作：权重（weights）以低精度格式存储（如 INT4、NF4、INT2、FP4），激活值（activations）保持较高精度（如 FP16），在 kernel 执行时动态将低精度权重反量化（dequantize）到计算精度，再执行矩阵乘法。TileLang 支持多种 dequantized GEMM 方案：W_INT2 × A_INT8（最高 7.65× over cuBLAS FP16），W_INT4 × A_FP16（1.04× over Marlin），W_NF4 × A_FP16（1.62× over BitsandBytes），FP4_E2M1 × FP16。TileLang 的关键优势在于通过寄存器内反量化（in-register dequantization）消除 Triton 的 shared memory layout conversion 瓶颈。

从算法 pipeline 角度拆解术语，比如术语所在 pipeline 的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

TileLang Dequantized GEMM (FP4_E2M1 × FP16) 算法 pipeline（图 17）：
```
输入: A[M, K] f16 (activation), B[N, K] u8 packed (4-bit weight, 2 elems per byte)
输出: C[N, M] f16

// === Step 1: 内存分配 ===
A_shared  = T.alloc_shared([block_M, block_K], f16)         // activation tile (shared mem)
B_shared  = T.alloc_shared([block_K, block_N//2], u8)       // packed weight tile (shared mem)
B_local   = T.alloc_fragment([block_N, block_K], u8)        // 解包前 (registers)
B_deq     = T.alloc_fragment([block_N, block_K], f16)       // 解包后 (registers)
Ct_local  = T.alloc_fragment([block_N, block_M], f32)       // accumulator (registers)

// === Step 2: Pipelined 主循环 ===
T.clear(Ct_local)
for k in T.Pipelined(K // block_K, num_stages=2):
  // Step 2a: 异步加载 activation 和 packed weight 到 shared memory
  T.copy(A[by*block_M, k*block_K], A_shared)               // cp.async global→shared
  T.copy(B[bx*block_N, k*block_K // 2], B_shared)          // B 每 byte 含 2 个 FP4 元素

  // Step 2b: Shared → Register (加载 packed weight)
  T.copy(B_shared, B_local)                                 // u8 连续字节 → registers

  // Step 2c: 寄存器内解量化 (FP4 → FP16)
  for i, j in T.Parallel(block_N, block_K):
    // _tir_packed_to_unsigned_convert: 从 u8 字节提取指定半字节 → unsigned int → cast to f16
    B_deq[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
      num_bits=4,                     // 每个 weight element 4 bits
      B_local[i, j // 2],             // 源 u8 字节 (含 2 个 FP4)
      j % 2,                          // 选择高/低 4-bit
      dtype=f16)

  // Step 2d: Tensor Core GEMM (transpose B)
  T.gemm(B_deq, A_shared, Ct_local, transpose_B=True)
  // B_deq^T [block_K, block_N] × A_shared^T [block_K, block_M] → Ct_local [block_N, block_M]

// === Step 3: 写出结果 ===
T.copy(Ct_local, Ct[bx*block_N, by*block_M])
```

关键优化对比：
- Triton 方式: unpack in registers → store to shared memory for layout conversion → ldmatrix reload → MMA（额外 shared memory 往返）
- TileLang 方式: load u8 in registers → View 零开销 reinterpret (u8→i4) → Cast vectorize to f16 → MMA（全程寄存器内完成，消除 shared memory 往返）
- TileLang 还支持 PTX 级 fast precision conversion 指令和 Ladder 的平滑内存访问优化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源代码：TileLang weights-only quantization GEMM kernel 通过 BitBLAS-TileLang 后端实现（BitBLAS 原是 TensorIR 后端，论文中替换为 TileLang 后端做对比）。同一程序模板通过参数化支持 INT2/INT4/NF4/FP4 等多种格式，仅需改变 num_bits 和 dtype 参数。在 A100 上，W_INT2A_INT8 达 7.65× cuBLAS FP16 speedup。对于不支持的量化格式，用户可自定义 _tir_packed_to_unsigned_convert 等 utility 函数。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
