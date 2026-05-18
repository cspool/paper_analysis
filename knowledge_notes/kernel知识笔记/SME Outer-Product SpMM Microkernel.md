## SME Outer-Product SpMM Microkernel

术语是什么？

SME Outer-Product SpMM Microkernel 是 ASM-SpMM 的核心 kernel，利用 ARM SME 的 MOPA outer-product 指令将稀疏矩阵-稠密矩阵乘法（SpMM）映射到 ZA tile 的外积累加。与传统的 CPU SpMM kernel（使用 SVE/Neon SIMD 做 vector inner-product/dot-product）不同，该 kernel 利用 outer-product 语义直接形成 sparse vector × dense tile 的二维外积并累加到 ZA，避免将 SME 降级为普通 SIMD 使用。Kernel 同时整合多 ZA tile 并发、显式 prefetch pipeline 来隐藏 SME 指令的高延迟和稀疏访存的不规则性。

从 kernel 调度角度拆解术语：

SME outer-product SpMM microkernel 的伪代码执行流（FP64, SVL=512, 8 rows/window）：
```
// 输入：OP-MCF formatted sparse matrix A, dense matrix B
// 输出：C = A × B，row_window_size = SVL/64 = 8 rows

for each row_window_r in 0..num_row_windows:
    // Phase 1: Clear and prepare
    svzero_za()                                  // 清零所有 ZA tile
    z_reg_pool = allocate_Z_registers(4)          // 4 Z regs for operand streaming
    
    // Phase 2: Process compressed slots with multi-tile concurrency
    for each compressed_slot_s in row_window_r:
        // ---- 当前 slot 的计算 ----
        col = SparseAtoB[s]
        pred = ColumnPositionMaskBit[s]           // bitmask→predicate
        
        sparse_vec = svld1_f64(z_reg[0], A_values[s])    // 加载 sparse values
        B_tile = svld1_f64(z_reg[1], &B[col*ldb])         // 加载 B 的对应 tile
        
        // outer product accumulate: sparse_vec ⊗ B_tile → ZA_tile[slot % num_tiles]
        ZA[slot % 4] = svmopa_za64_f64_m(ZA[slot % 4], pred, all_true, 
                                          sparse_vec, B_tile)
        
        // ---- 预取下一 slot（与当前计算 overlap）----
        _svprfw(A_values[s+1])                    // 预取 sparse data
        _svprfw(&SparseAtoB[s+1])                 // 预取 column index
        _svprfw(&ColumnPositionMaskBit[s+1])      // 预取 bitmask
        _svprfw(&B[SparseAtoB[s+1] * ldb])        // 预取 dense B tile
    
    // Phase 3: Writeback
    for each active_ZA_tile:
        svst1_hor_za64_f64(C[row_window_r + tile_offset], ZA_tile)
```

关键执行特征：
1. **Outer-product 语义**：sparse_vec[i] × B_tile[j] 直接产生二维结果→累加到 ZA[i][j]。无中间 dot-product reduction
2. **Predicate mask 控制**：pred 确保只有有效非零行参与 outer product，零行对应的 ZA row 保持原值
3. **多 tile 流水线**：4 个 ZA tile 轮转使用，当前 tile 计算时前一 tile 结果的写回可与下一 slot 的加载重叠
4. **Prefetch 策略**：_svprfw 预取指令提前将下一 slot 的 sparse data/mask/column index/dense B 片段带入 L2 cache，将 LLC miss rate 从 30%-61% 降至 23%-48%

术语一般如何实现？如何使用？

实现依赖 ARM SME intrinsics（`arm_sme.h`）和 streaming SVE 模式。编译需 Clang 16.0+（支持 SME intrinsics 和 `-march=armv9-a+sme`）。Apple M4 上 key parameter：SVL=512→row window=8 rows（FP64），ZA 可划分 8 个独立 tile。使用注意：(1) SME 指令延迟高（M4 上 10-20 cycles），必须多 tile 并发+prefetch 隐藏；(2) ZA tile 间无直接数据通路，跨 tile 累加需 Z register 中转；(3) 对非常稀疏的 block（per-slot NNZ 接近 1），outer-product 的 tile 利用率低→应分配 SVE/NEON vector path。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

