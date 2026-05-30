## Quantization Inefficiency (GEMM Scheduling)

术语是什么？
Quantization inefficiency（量化低效）是GPU GEMM调度中的一种处理器利用率损失现象：当output tile数量不能被SM数量整除时，最后部分wave中的CTA数量少于SM数量，导致部分SM空闲等待，无法达到理论峰值吞吐。例如，384×384×128 GEMM在BLK_M=128下产生3×3=9个output tile，在4-SM GPU上需要3波执行（wave 1: 4 CTA, wave 2: 4 CTA, wave 3: 1 CTA + 3 SM idle）→ 利用率上限75%。该问题随GPU SM数量的增加而加剧（更宽的处理器意味着更少的wave，更大的最后一波partial wave比例），也随blocking factor增大而加剧（更大的tile→更少的tile→更少的wave）。常见GEMM workload中最后一个部分full wave可能占总计算时间的显著比例。

从kernel调度角度拆解术语：
量化低效的定量分析：

```
给定: m×n×k GEMM problem, BLK_M, BLK_N, 处理器有p个SM
Number of output tiles: t = ceil(m/BLK_M) × ceil(n/BLK_N)
Number of full waves: w_full = floor(t/p)
Number of remaining tiles: r = t - w_full × p
Utilization ceiling: 
  - if r == 0: 100%
  - if r > 0: (w_full × p + r) / ((w_full + 1) × p) × 100%

Example:
  m=n=384, k=128, BLK_M=BLK_N=128 → t=9, p=4
  w_full = 2, r = 1
  Utilization ≤ 9/12 = 75%

  m=n=384, BLK_M=BLK_N=64 → t=36, p=4
  w_full = 9, r = 0
  Utilization = 100% (但更小的tile意味着更低的cache/scratchpad效率)
```

解决方案：(1) Ensemble of tiling configurations——cuBLAS/CUTLASS提供多种blocking factor，通过heuristics选择量化效果最优的配置；(2) Fixed-split——沿k轴split tile增加CTA数；(3) Stream-K——以MAC-loop iteration为单位分配，天然避免量化低效。

术语一般如何实现？如何使用？
量化低效是tile-based并行分解的固有特征——任何将work quantum定为output tile的方法都会在tile数与处理器宽度的关系上产生离散化损耗。解决此问题需要改变work量子化粒度（Stream-K的MAC-loop iteration）或引入tile-splitting（fixed-split、Stream-K的更泛化形式）。Stream-K通过将量子化单位缩小32-512×（取决于⌈k/BLK_K⌉），使量化低效在实际上可以忽略。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
