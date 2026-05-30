## Hybrid Stream-K + Data-parallel Decomposition

术语是什么？
Hybrid Stream-K + Data-parallel decomposition是Stream-K的实际生产实现，结合了data-parallel的cache-friendly tile-aligned执行和Stream-K的workload balancing。其核心思想是：对大多数output tile使用data-parallel执行（完整wave、无tile-processing skew、良好cache reuse），仅对最后部分data-parallel wave的剩余tile应用Stream-K的iteration balancing。具体而言，完整的data-parallel wave数w = ⌊t/p⌋（t为output tile数，p为SM数），然后提前终止1个完整wave，将剩余tile（p到2p-1个tile的MAC iterations）均匀分配给g=p个CTA，确保每CTA获得1-2个tile的iteration量。

从kernel调度角度拆解术语：
"Two-tile Stream-K + data-parallel"混合调度的执行schedule（Figure 3c）：

```
假设: 896×384×128 GEMM, BLK_M=128, BLK_N=128, BLK_K=32, p=4
t = ceil(896/128) × ceil(384/128) = 7×3 = 21 output tiles
w = floor(21/4) = 5 full data-parallel waves
→ 执行 w-1 = 4 个完整data-parallel wave (16 tiles)
→ 剩余 5 tiles 的MAC iterations均匀分配给 4 CTA
   iters_per_cta = (5 × ceil(128/32)) / 4 = 5×4/4 = 5 MAC-loop iterations
  
CTA调度:
  Wave 1-4: 4×4=16个CTA, data-parallel, 每CTA完整处理1个tile
  Wave 5 (Stream-K):
    CTA_0: tile_16 iter 0-4 → StoreTile(tile_16)
    CTA_1: tile_16 iter 4-8 → StorePartials(tile_17 partial)
            tile_17 iter 8-12 → StorePartials(tile_17 partial)
            tile_17 iter 12-16 → StoreTile(tile_17)
    CTA_2: tile_18 iter 0-4 → StoreTile(tile_18)
    CTA_3: tile_19 iter 0-4 → StoreTile(tile_19)
            tile_20 iter 4-8 → StoreTile(tile_20)

注意: tile_17跨越多个CTA边界, 但每tile最多2个CTA覆盖
      tile_16/18/19/20只需1个CTA (tile-aligned)
```

优点：(1)大多数tile无synchronization/partial sum overhead；(2)cache locality在data-parallel wave中得到保持；(3)Stream-K wave中每tile最多2个CTA覆盖（vs 基本Stream-K可能更多）→ 同步延迟有效隐藏；(4)与基本Stream-K共用同一kernel实例。

术语一般如何实现？如何使用？
混合调度在CUTLASS的Stream-K实现中通过修改runtime grid size和iteration分配逻辑完成，使用与基本Stream-K相同的kernel模板（Algorithm 5）。它实现了更好的性能稳定性——在memory-bound regime中cache locality更重要，在compute-bound regime中Stream-K的负载均衡更重要。混合调度是对基本Stream-K在cache性能上的补充优化。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
