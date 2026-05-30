## Fixed-split GEMM Decomposition

术语是什么？
Fixed-split GEMM decomposition是data-parallel的一种扩展：对于每个output tile，将沿k轴的MAC-loop迭代范围split为s份，分配s个CTA合作计算。每个CTA执行⌈⌈k/BLK_K⌉/s⌉个MAC-loop迭代，然后通过fixup步骤（partial sum通信和累积）合并结果。当s=1时，fixed-split退化为data-parallel。CUTLASS和cuBLAS均包含fixed-split实现。splitting factor s是runtime参数，允许单个kernel支持多种work volumes。

从kernel调度角度拆解术语：
Fixed-split GEMM的伪代码（Algorithm 4）：

```
iters_per_tile = ceil(k/BLK_K)
iters_per_split = ceil(iters_per_tile / s)

// Grid: ceil(m/BLK_M) × ceil(n/BLK_N) × s 个CTA
FOR EACH CTA_{[x, y]}, x=tile_idx, y=split_idx:
    mm = BLK_M * (x / ceil(n/BLK_N))
    nn = BLK_N * (x % ceil(n/BLK_N))
    iter = y * iters_per_split
    iter_end = min(iters_per_tile, iter + iters_per_split)
    
    // 执行分配的MAC迭代范围
    accum = MacLoop(x, iter, iter_end)
    
    IF y != 0:
        // 非首个split: 写partial sums到temporary global storage
        StorePartials(partials[x, y], accum)
        Signal(flags[x, y])
    ELSE:
        // 首个split (y=0): 等待并累积其他split的partials
        FOR cta IN [1, s):
            Wait(flags[x, cta])
            accum += LoadPartials(partials[x, cta])
        StoreTile(C, x, accum)
```

Fixed-split的fixup overheads随问题和splitting factor增长：每个tile需要s-1次额外的partial sum读写和同步。当s=32时（类比Stream-K的量化粒度），fixup overhead增至8×。

术语一般如何实现？如何使用？
Fixed-split在CUTLASS中通过runtime splitting factor参数实现。相比data-parallel，它通过沿k轴增加并行度减小了量化低效的影响——更多的CTA减少了对output tile数量均匀性的依赖。然而，由于s对所有tile统一应用，它不太可能为任意问题shape达到完美量化（Figure 2a展示了某shape上即使s=2也只能达到90%效率）。它也无法享受Stream-K的自动时间偏移同步隐藏（tile-processing skew），因为同一tile的所有split CTA在相近时间完成计算。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
