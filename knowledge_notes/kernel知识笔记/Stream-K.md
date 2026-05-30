## Stream-K

术语是什么？
Stream-K是一种work-centric的GEMM并行分解策略，由Muhammad Osama、Duane Merrill等人在PPoPP 2023上提出。其核心理念是将MAC-loop迭代（而非传统的output tile）作为跨GPU SM的workload量子化单元。具体而言：给定GEMM问题m×n×k，总MAC-loop迭代次数total_iters = ⌈m/BLK_M⌉ × ⌈n/BLK_N⌉ × ⌈k/BLK_K⌉，Stream-K将这total_iters次迭代均匀分配给g个CTA（Cooperative Thread Arrays），每个CTA执行⌈total_iters/g⌉个连续的MAC-loop迭代，迭代范围沿m→n→k线性化映射，可跨越output tile边界。当一个CTA的起始/结束迭代不与tile边界对齐时，通过temporary global storage交换partial sums，由覆盖该tile的k=0迭代的CTA负责累积并写出最终结果。

与data-parallel分解（将output tiles分配给CTA，量化单位为整个tile，通常32-512个MAC iterations）相比，Stream-K的量化单位（1个MAC-loop iteration = BLK_M×BLK_N×BLK_K MACs）精细32-512倍，因此量化效率（quantization efficiency）可接近100%。Stream-K的communication/synchronization/global storage overheads仅与CTA数g（≈处理器SM数p）成正比（O(p)），与问题规模无关。当output tile数大于CTA数时，每个tile最多被2个CTA覆盖，且tile-processing的时间偏移自然隐藏inter-CTA同步等待。

Stream-K也泛化到fixed-split和data-parallel分解：当g为output tile数的整倍数时，等价于fixed-split；当g=output tile数时，等价于data-parallel。

从kernel调度角度拆解术语：
Stream-K的核心算法（Algorithm 5）如下（伪代码）：

```
// 输入：m×n×k GEMM问题，blocking factors BLK_M/BLK_N/BLK_K，grid size g
total_iters = ceil(m/BLK_M) * ceil(n/BLK_N) * ceil(k/BLK_K)
iters_per_cta = ceil(total_iters / g)
iters_per_tile = ceil(k/BLK_K)

// 启动 g 个 CTA
FOR EACH CTA_x, x ∈ [0, g):
    iter = x * iters_per_cta
    iter_end = min(total_iters, iter + iters_per_cta)
    
    WHILE iter < iter_end:
        tile_idx = iter / iters_per_tile
        tile_iter = tile_idx * iters_per_tile
        tile_iter_end = tile_iter + iters_per_tile
        
        // 在tile内执行的MAC迭代范围
        local_iter = iter - tile_iter
        local_iter_end = min(iter_end, tile_iter_end) - tile_iter
        
        // 执行MAC-loop计算
        accum = MacLoop(tile_idx, local_iter, local_iter_end)
        
        // 判断当前CTA是否覆盖tile的起始/结束
        tile_started = (iter == tile_iter)
        tile_ended = (iter_end >= tile_iter_end)
        
        IF NOT tile_started:
            // 从其他CTA接收partial sums
            StorePartials(partials[x], accum)
            Signal(flags[x])
            IF NOT tile_ended:
                // 累积其他CTA的贡献
                FOR cta IN [x+1 .. tile_iter_end/iters_per_cta]:
                    Wait(flags[cta])
                    accum += LoadPartials(partials[cta])
            StoreTile(C, tile_idx, accum)
        ELSE:
            IF tile_ended:
                StoreTile(C, tile_idx, accum)
            // 否则partial sums将在下一while迭代或另一CTA中处理
        
        iter = tile_iter_end
```

CTA内的MacLoop()子程序执行指定范围的MAC-loop迭代，每迭代执行BLK_M×BLK_N×BLK_K次MAC操作，使用shared memory进行两级blocking（global→shared→register），包含software pipelining隐藏global/shared memory延迟。

术语一般如何实现？如何使用？
Stream-K已在NVIDIA CUTLASS 2.11中开源（https://github.com/NVIDIA/cutlass）。实现关键点：
1. **Grid size选择**：使用解析模型CTA_time(g) = a + b·(FixupPeers(g)>1) + c·ItersPerCta(g) + d·(FixupPeers(g)-1)，其中a为固定开销（launch latency, cold cache misses, final tile write），b为partial sum输出的条件开销，c为每MAC迭代的指令/stall开销，d为每协作CTA的partial sum累积开销。参数{a,b,c,d}通过微基准一次测量per architecture确定。
2. **混合调度**：实现"two-tile Stream-K + data-parallel"混合调度：仅对最后部分data-parallel wave的剩余tile进行iteration balancing，确保每个tile最多被2个CTA覆盖。
3. **单kernel per precision**：vs cuBLAS的20+ kernel specialization，仅需每个精度一个kernel，可执行代码减少约20×。
4. **性能**：在NVIDIA A100上，FP16→32 Stream-K比data-parallel CUTLASS平均快1.63×（最大14.7×），比cuBLAS平均快1.13×（最大6.74×），跨32,824个GEMM shapes评估。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
