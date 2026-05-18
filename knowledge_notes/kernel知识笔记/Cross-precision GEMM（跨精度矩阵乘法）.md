## Cross-precision GEMM（跨精度矩阵乘法）

术语是什么？通过联网搜索让回答具体和精准。

Cross-precision GEMM（跨精度通用矩阵乘法）是指在一次矩阵乘法C=A×B中，操作数A和B使用不同数值精度的计算模式。在RoMeo的混合精度量化场景中，weight和activation可能各自处于INT4或INT8精度，产生四种精度组合：W4A4（weight INT4 × activation INT4）、W4A8（weight INT4 × activation INT8）、W8A4（weight INT8 × activation INT4）、W8A8（weight INT8 × activation INT8）。这四种组合的计算吞吐量不同：INT4 Tensor Core提供~8×FP16峰值吞吐，INT8提供~2×FP16。Cross-precision GEMM使用INT32累加器防止混合精度累加时overflow。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RoMeo中cross-precision GEMM的separate-kernels实现（以Qwen3-8B Down_proj [8192×4096×4096]为例，batch=64）：

```
// 四种separate kernel，各自操作dense uniform-precision矩阵
// 每个kernel使用CUTLASS实现，内部含software pipeline

Kernel W4A4: // 主体（~88%计算量），最高吞吐
  Input:  A_int4 [M_normal, K] packed, B_int4 [K, N_normal] packed
  // 使用m16n8k32 tiling, INT4 Tensor Core
  for tile_k in 0..K/32:
    cp.async A_tile[tile_k] → smem_A
    cp.async B_tile[tile_k] → smem_B  
    cp.async.wait(Nstage-1)
    mma.sync.aligned.m16n8k32 smem_A, smem_B → accum
  Output: C_W4A4 [M_normal, N_normal] FP32

Kernel W4A8: // activation outlier rows × normal weight cols
  Input: A_int8 [M_outlier, K], B_int4 [K, N_normal] packed
  // INT4×INT8 cross-precision: B在SMEM内cast到INT8
  for tile_k in 0..K/32:
    cp.async A_tile, B_tile → smem
    // INT4 → INT8 casting in shared memory
    // 使用两条binary arithmetic指令而非type conversion
    smem_B_int8 = cast_int4_to_int8(smem_B_int4)  
    cp.async.wait(Nstage-1)
    mma.sync.aligned.m16n8k16 smem_A_int8, smem_B_int8 → accum  
  Output: C_W4A8 [M_outlier, N_normal] FP32

Kernel W8A4: // normal activation rows × outlier weight cols
  // 类似W4A8，A在SMEM内cast

Kernel W8A8: // outlier交叉部分（~0.25%计算量，但需更高精度）
  Input: A_int8 [M_outlier, K], B_int8 [K, N_outlier]
  // 纯INT8 GEMM，使用INT8 Tensor Core
  // Shared memory需求为INT4-INT4 kernel的2倍
```

关键设计选择separate-kernels而非fused-kernel：
- INT4-INT4 kernel：shared memory小→compiler可用更多register做loop unrolling提升ILP
- INT8-INT8 kernel：shared memory大→occupancy由SMEM限制，compiler自动减少register使用
- Fused kernel无法为不同精度组合独立分配on-chip资源→suboptimal

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- CUTLASS模板库提供INT4/INT8 GEMM的参考实现，RoMeo在CUTLASS基础上customize supporting mixed input types
- 类型转换策略：在shared memory内用binary arithmetic（IADD3+LOP3）完成INT4→INT8 casting，避免昂贵的PTX type conversion指令
- Software pipeline（Algorithm 2 in RoMeo）：使用cp.async异步GMEM→SMEM加载→pipeline fill (Nstage iterations)→steady state (wait+mma+issue)→drain
- 异步并发执行：四种kernel在不同CUDA stream上并发执行→掩盖单独kernel的launch overhead和tall-and-skinny矩阵的SM underutilization
- 结果合并：低精度W4A4结果为基础，高精度outlier计算结果通过post-mul overwrite覆盖对应位置

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

