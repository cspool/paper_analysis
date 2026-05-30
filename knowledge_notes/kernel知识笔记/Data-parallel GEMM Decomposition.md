## Data-parallel GEMM Decomposition

术语是什么？
Data-parallel GEMM decomposition是GPU上经典的GEMM并行化策略：将输出矩阵C划分为BLK_M×BLK_N的tile，每个CTA独立计算一个完整的output tile。Grid包含⌈m/BLK_M⌉ × ⌈n/BLK_N⌉个CTA，每个CTA对分配的tile执行完整的⌈k/BLK_K⌉个MAC-loop迭代（沿k轴递进累积）。CTA之间无通信——各output tile的计算完全独立。这是CUTLASS和cuBLAS中GEMM kernel的基础分解方式。

从kernel调度角度拆解术语：
Data-parallel GEMM的伪代码（Algorithm 2+3）：

```
// Grid: ceil(m/BLK_M) × ceil(n/BLK_N) 个CTA
FOR EACH CTA_{[mm_blk, nn_blk]}:
    mm = mm_blk * BLK_M
    nn = nn_blk * BLK_N
    accum[BLK_M, BLK_N] = 0
    
    // 沿k轴累积
    FOR kk IN [0, k) STEP BLK_K:
        // Load A tile: [mm:mm+BLK_M, kk:kk+BLK_K]
        frag_a = LoadFragment(A, mm, kk)
        // Load B tile: [kk:kk+BLK_K, nn:nn+BLK_N]
        frag_b = LoadFragment(B, kk, nn)
        
        // MAC iteration (fully unrolled per thread)
        FOR EACH THREAD_{[mmm, nnn]} IN [BLK_M, BLK_N]:
            FOR kkk IN [0, BLK_K):
                accum[mmm, nnn] += frag_a[mmm, kkk] * frag_b[kkk, nnn]
    
    // 写出output tile
    StoreTile(C, [mm, nn], accum)
```

CTA（Cooperative Thread Array，即thread block）被GPU SM调度器以"wave"形式dispatch。当output tile数不能被SM数整除时，最后部分wave中部分SM空闲，造成量化低效（quantization inefficiency）。例如384×384×128问题，9个128×128 tile在4-SM GPU上需要3波（4+4+1），利用率上限75%。

术语一般如何实现？如何使用？
在CUTLASS中，data-parallel GEMM通过多层模板抽象实现，包括：threadblock-level tiling、warp-level tiling、thread-level register blocking、software pipelining of shared memory data movement。CUTLASS提供多种blocking factor specialization（例如FP64: 32×32×16、32×64×16、64×64×16、64×128×16、128×128×16；FP16→32: 64×64×64、64×128×32、128×128×32、128×256×32）。cuBLAS为每种精度提供20+个data-parallel和fixed-split kernel variant，通过复杂heuristics或ML模型选择kernel。在data-parallel分解中，每个CTA的工作量与k维大小成正比（⌈k/BLK_K⌉个MAC-loop iterations），因此当m×n小而k大时（强伸缩scenario），单个CTA的工作量极端不平衡于其他CTA（其他CTA空闲），造成显著的性能损失。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
