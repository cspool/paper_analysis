# B.2 Dequantized Matrix Multiplication

```
1 @tilelang.jit
2 def matmul_fp16_fp4(
3 A: T.Tensor(A_shape, in_dtype),
4 B: T.Tensor(B_shape, storage_dtype),
5 Ct: T.Tensor((N, M), out_dtype),
6 ):
7 with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
8 A_shared = T.alloc_shared(A_shared_shape, in_dtype)
9 B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
10 B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
11 B_dequantize_local = T.alloc_fragment(B_dequantize_shared_shape, in_dtype)
12 Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
13
14 T.clear(Ct_local)
15 for k in T.Pipelined(
16 T.ceildiv(K, block_K),
17 num_stages=num_stages
18 ):
19 T.copy(A[by * block_M, k * block_K], A_shared)
20 T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
21 T.copy(B_shared, B_local)
22 for i, j in T.Parallel(block_N, block_K):
23 B_dequantize_local[i, j] = _tir_packed_to_unsigned_convert("int", 8)(
24 num_bits,
25 B_local[i, j // 2],
26 j % 2,
27 dtype=in_dtype,
28 )
29 T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)
30 T.copy(Ct_local, Ct[bx * block_N, by * block_M])
```

Fig. 17. Implementation of Weight-Only Quantization (FP4\_E2M1FP16) Matmul using TileLang, showcasing support for mixed-precision computations via a simple form.

