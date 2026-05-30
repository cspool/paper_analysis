## TileLang: A Composable Tiled Programming Model for AI Systems

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是TileLang支持的低精度/混合精度矩阵乘法（Dequantized Matmul），覆盖多种量化方案：(1) Weight-Only Quantization Matmul: W_INT2 × A_INT8, W_INT4 × A_FP16, W_NF4 × A_FP16；(2) 混合精度FlashAttention: FP16 compute with FP32 accumulation, online softmax rescaling；(3) Multi-Head Latent Attention (MLA): KV cache压缩下的高效attention计算；(4) Linear Attention: Mamba-2 chunk-scan/chunk-state函数的高效实现。

  实验比较：(1) Dequantized Matmul (A100): W_INT2A_INT8 7.65× over cuBLAS-W_FP16A_FP16; W_INT4A_FP16 平均1.04× over Marlin; W_NF4A_FP16 平均1.62× over BitsandBytes。(2) FlashAttention (H100): 1.36× over FA3, 1.41× over Triton, 1.70× over PyTorch。(3) Linear Attention (H100): 平均1.77×和2.10× over Triton (chunk-scan和chunk-state)。(4) MLA (H100): 1075.9× over Torch, 98% of FlashMLA。(5) GEMM (RTX 4090/A100/H100/MI300X): FP16/F32 precision, 0.97-1.10× vs vendor libraries。

- 硬件平台是什么，配置是什么。
  Dequantized Matmul: NVIDIA A100 (80 GB, Ampere, CUDA 12.4)。FlashAttention/Linear Attention/MLA: NVIDIA H100 (80 GB, Hopper, CUDA 12.4)。MLA: AMD Instinct MI300X (192 GB, ROCm 6.1.0)。GEMM: RTX 4090, A100, H100, MI300X。所有平台Ubuntu 20.04。

- 模型是什么。数据集和bench分别是什么。
  算子级benchmark（非end-to-end模型推理），覆盖的算子来自大模型典型workload：
  - GEMM: Table 2的16种矩阵shape（M ∈ [1, 8192], N ∈ [1024, 57344], K ∈ [8192, 57344]），覆盖不同矩阵乘法问题的尺寸
  - FlashAttention: Table 3的5种配置（batch=1, nheads=32, seq_len=512/1024/4096, head_dim=128, causal/non-causal）
  - Linear Attention: Table 4的12种配置（chunk-scan CC0-CC5和chunk-state CT0-CT5, batch=1/64, nheads=64, seq_len=1024/2048/8192, head_dim=64, d_state=128）
  - MLA: 论文未列出具体MLA benchmark形状的详细表格（主要展示性能speedup和代码行数对比）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/tile-ai/tilelang

  算法pipeline —— Dequantized Matmul (FP4_E2M1 × FP16, 对应图17):
  ```
  输入: A[K,M] f16 (activation), B[K,N] u8 packed (4-bit weight, 2 elems per byte)
  输出: C[N,M] f16

  伪代码（Python TileLang程序）:
  @tilelang.jit
  def matmul_fp16_fp4(A: T.Tensor, B: T.Tensor, Ct: T.Tensor):
    with T.Kernel(N//block_N, M//block_M, threads=threads) as (bx, by):
      # 1) 内存分配
      A_shared = T.alloc_shared([block_M, block_K], f16)         # shared mem: activation tile
      B_shared = T.alloc_shared([block_K, block_N//2], u8)       # shared mem: packed weight tile
      B_local  = T.alloc_fragment([block_N, block_K], u8)        # registers: weight
      B_deq    = T.alloc_fragment([block_N, block_K], f16)       # registers: dequantized weight
      Ct_local = T.alloc_fragment([block_N, block_M], f32)       # registers: accumulator

      T.clear(Ct_local)
      # 2) Pipelined主循环
      for k in T.Pipelined(K // block_K, num_stages=2):
        # Step a: 异步加载activation和packed weight到shared memory
        T.copy(A[by*block_M, k*block_K], A_shared)
        T.copy(B[bx*block_N, k*block_K // 2], B_shared)         # B每byte含2个FP4

        # Step b: 从shared memory到register
        T.copy(B_shared, B_local)

        # Step c: 寄存器内解量化 (FP4 → FP16)
        for i, j in T.Parallel(block_N, block_K):
          B_deq[i,j] = _tir_packed_to_unsigned_convert("int", 8)(
            num_bits=4, B_local[i, j//2], j%2, dtype=f16)
          # 从u8字节中提取高/低4-bit (j%2选择) → 转为unsigned int8 → cast to float16

        # Step d: Tensor Core矩阵乘法
        T.gemm(B_deq, A_shared, Ct_local, transpose_B=True)
        # B_deq[block_N, block_K]^T × A_shared[block_M, block_K]^T 累加到 Ct_local[block_N, block_M]

      # 3) 写出结果
      T.copy(Ct_local, Ct[bx*block_N, by*block_M])
  ```

  张量计算流程（单tile）:
  - A activation tile: f16 [block_M, block_K]  global→shared (cp.async)
  - B weight tile:    u8  [block_K, block_N/2]  global→shared (cp.async, pipelined)
  - Dequant: u8 [block_N, block_K] → f16 [block_N, block_K] (in-register, per element)
  - MMA: B_deq^T [block_K, block_N] × A_shared^T [block_K, block_M] → Ct_local [block_N, block_M] (Tensor Core f32 accumulate)
  - Ct_local → Ct[bx*block_N : (bx+1)*block_N, by*block_M : (by+1)*block_M] (register→global, f16 store)

  关键优化点（与Triton的区别）：
  - 权重以packed u8形式直接在shared memory存储，无需shared memory上的layout conversion（Triton需要将解包后的register tensor layout通过shared memory转换到Tensor Core兼容格式）
  - 解量化（dequantize）在寄存器内完成，配合View零开销类型reinterpret
  - Pipeline自动overlap weight/activation loading与computation
  - 对于INT2/INT4/NF4格式，TileLang可由同一程序模板参数化生成，仅需改变num_bits和dtype

  **近似层次匹配说明**：TileLang本身是编译框架/kernel调度工具，但其支持的Dequantized Matmul、FlashAttention、Linear Attention、MLA属于算法pipeline层面的低精度/高效attention算法实现。按最接近层次分类到算法pipeline。
