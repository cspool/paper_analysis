# B.1 Matrix Multiplication (Matmul)

```
1 @tilelang.jit
2 def Matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):
3 with T.Kernel(N // block_N, M // block_M,
4 threads=threads) as (bx, by):
5 A_shared = T.alloc_shared(block_M, block_K)
6 B_shared = T.alloc_shared(block_K, block_N)
7 C_local = T.alloc_fragment(block_M, block_N)
8
9 T.clear(C_local)
10 for k in T.Pipelined(K // block_K, num_stages=2):
11 T.copy(A[by * block_M, k * block_K], A_shared)
12 T.copy(B[k * block_K, bx * block_N], B_shared)
13 T.gemm(A_shared, B_shared, C_local)
14
15 T.copy(C_local, C[by * block_M, bx * block_N])
```

Fig. 16. Kernel Implementation of Matrix Multiplication.

