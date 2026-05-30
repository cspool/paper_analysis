## MAC-loop Iteration (GEMM)

术语是什么？
MAC-loop iteration是GPU GEMM kernel中单次CTA-wide的multiply-accumulate迭代。一个MAC-loop iteration的计算量为BLK_M×BLK_N×BLK_K次MAC操作（其中BLK_M、BLK_N是output tile的高宽，BLK_K是accumulation维度上的tile大小）。每个MAC-loop iteration内部包含：(1) 从global memory加载A tile（BLK_M×BLK_K）和B tile（BLK_K×BLK_N）到shared memory；(2) 从shared memory加载fragments到寄存器；(3) 每个线程执行(BLK_M×BLK_N×BLK_K)/CTA_THREADS次MAC操作（fully unrolled）；(4) 通过software pipelining使data movement与MAC操作重叠。

在Stream-K中，MAC-loop iteration被用作跨SM的workload量子化单位。单个iteration的MAC量远小于整个output tile（后者=⌈k/BLK_K⌉个MAC-loop iterations），因此以iteration为单位的partition可实现更精细的负载均衡。

从kernel调度角度拆解术语：
以FP16→32 GEMM, BLK_M=128, BLK_N=128, BLK_K=32, CTA_THREADS=256为例：

```
一个MAC-loop iteration:
  1. TMA/LDS加载A tile: A[128×32] (128×32×2 bytes = 8KB, FP16)
  2. TMA/LDS加载B tile: B[32×128] (8KB, FP16)
  3. 每个线程的寄存器计算量:
     (128×128×32) / 256 = 2048 MACs/thread/iteration
     每线程维护 128×128/256 = 64 个output元素的累加器
  4. Tensor Core WGMMA: 
     mma.m16n8k16 × N 次 → 每个warp 128×128输出
  5. Software pipeline: 当前iteration的MMA与下个iteration的TMA load重叠
  
一个output tile的MAC-loop iterations = ceil(128/32) = 4 (k=128)
完整GEMM (384×384×128): 
  total_iters = 3×3×4 = 36 MAC-loop iterations
  vs 9 output tiles (data-parallel量子化单位)
  → MAC-loop iteration粒度细4×
```

术语一般如何实现？如何使用？
在CUTLASS中，MAC-loop iteration通过MacLoop()子程序实现（Algorithm 3），该子程序封装了CTA-wide的shared memory staging和per-thread Tensor Core MMA。MacLoop()接收起始和结束iteration index，执行指定范围的MAC-loop iterations。在data-parallel和fixed-split中，参数为tile内的iteration范围（0到⌈k/BLK_K⌉或split子范围）；在Stream-K中，参数可能跨越tile边界。MAC-loop iteration也是Stream-K的grid size选择解析模型中per-iteration workload cost参数c的物理基础。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---
