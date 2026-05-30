# 知识库_kernel调度

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

## Tile-processing Skew

术语是什么？
Tile-processing skew是Stream-K分解的一种副作用：当output tile数t不能被grid size g整除时，各CTA的起始k-offset不同。例如在384×384×128问题上g=4 CTA、BLK_K=4、每CTA 72个MAC iterations时，CTA_0起始k=0、CTA_1起始k=288、CTA_2起始k=576等。这种offset差异持续整个GEMM计算过程（persistent skew），可能导致不同CTA加载的k-axis fragments来自不同的k位置，从而阻止这些fragments在GPU L2 cache中被跨CTA复用。

从kernel调度角度拆解术语：
Tile-processing skew与cache reuse的关系：

```
384×384×128 GEMM, BLK_K=32, 4 CTA:
  CTA_0: iter [0, 72)   → k∈[0, 288)
  CTA_1: iter [72, 144) → k∈[288, 576) [部分覆盖tile 2-4]
  CTA_2: iter [144, 216)→ k∈[576, 864)
  CTA_3: iter [216, 288)→ k∈[864, 1152) [但k max=128 所以wraparound]

此时各CTA在相同相对进度下访问不同的k-axis A/B fragments
→ L2 cache中的fragments难以被多个CTA复用
→ cache hit rate下降
```

解决方案："two-tile Stream-K + data-parallel"混合调度——减少Stream-K的应用范围，使大部分CTA执行完整的data-parallel wave，仅对最后wave的剩余tile做iteration balancing。这限制了skew的持续时间，使大部分执行在无skew的完整wave中进行，同时cache locality得到改善。

术语一般如何实现？如何使用？
混合调度（Section 5.2）通过以下策略控制skew：
- "Data-parallel + one-tile Stream-K"：仅将Stream-K应用于最后部分wave的tile
- "Two-tile Stream-K + data-parallel"：每CTA接收1-2个tile的iteration量，既限制skew，又确保每tile最多2个CTA覆盖（良好同步隐藏）
- 对于w≥2（至少2个完整wave），每accumulating CTA仅需接收1个其他CTA的partials
混合调度的实现使用与基本Stream-K相同的kernel结构，仅通过修改grid size和iteration分配逻辑来实现不同的调度策略。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---

## Grid Size Selection Heuristic (Stream-K)

术语是什么？
Stream-K的grid size选择启发式是一个解析模型，用于在启动kernel前确定最优的CTA数g，以最小化总执行时间。模型将单个CTA的运行时间建模为四项之和：CTA_time(g) = a + b·(FixupPeers(g)>1) + c·ItersPerCta(g) + d·(FixupPeers(g)-1)，其中：(a)固定开销——grid launch latency、compulsory cache misses、final output tile write；(b)partial sum输出条件开销——当tile数不能被完美量化时产生；(c)每MAC-loop iteration的指令/stall workload；(d)每协作CTA的partial sum读取和累积开销。ItersPerCta(g) = ⌈total_iters/g⌉，FixupPeers(g) = ⌈⌈k/BLK_K⌉/ItersPerCta(g)⌉为覆盖同一tile的CTA数。

从kernel调度角度拆解术语：
该模型的行为因问题shape而异：
- **大k、少量output tile**：reduction in MAC-loop时间单调优于fixup cost增长 → g_opt = p（全处理器宽度并行）
- **中等k、中等output tile**：fixup cost在g超过某个点后超过iteration减少收益 → g_opt < p（出现全局最小点）
- **极小m×n、极大k**（1个output tile）：虽然强伸缩潜力大，但per-peer serial reduction cost全部由单个CTA承担 → g_opt << p

参数{a,b,c,d}对每个(blocking factors, 数据类型, GPU架构)组合唯一，通过微基准经验测定，只需每个target architecture执行一次，然后将参数静态编译入库。

术语一般如何实现？如何使用？
与cuBLAS/CUTLASS的ensemble方法（静态生成20+ kernel variant + 运行时复杂heuristics/ML选择）不同，Stream-K的启发式模型：(1)仅需一次per-architecture微基准标定参数；(2)参数静态编译入库；(3)运行时快速评估模型选择g。这消除了对复杂kernel selection heuristics的需求，同时保持了动态适应性。模型也可判断何时退化为纯data-parallel（g = output tile数）是最优的。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

---

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

## CUDA Stream

术语是什么？
CUDA Stream 是 CUDA 编程模型中的 FIFO 命令队列，用于管理和排序 GPU operations（kernel launch 和 memory copy）。同一 stream 内的操作严格按 FIFO 顺序执行，不同 stream 间的操作可以并发执行（是否真正并发取决于 GPU scheduler 和硬件资源）。默认情况下，所有 GPU operations 被提交到单一的 default stream（NULL stream）。程序员可创建额外 stream 实现并发操作。

从kernel调度角度拆解术语：
在本文发现的 TX2 GPU scheduler 模型中，每个 CUDA stream 对应一个 stream queue（FIFO）。GPU operation 的生命周期为：
1. CUDA API 调用时入队到对应 stream queue（Rule G1）
2. Kernel 到达 stream queue 头部时入队到 EE queue（Rule G2）；Copy 到达 stream queue 头部时入队到 CE queue（Rule C1）
3. EE queue 头部的 kernel 的 block 被分配到 SM（受资源约束和优先级规则限制）
4. Kernel 所有 block 完成后从 EE queue 出队（Rule G3），随后从 stream queue 出队（Rule G4）

伪代码——单个 stream 的 operation 流转：
```
// CPU 端: CUDA 程序
cudaStream_t s1;
cudaStreamCreate(&s1);

// Operation enqueue (Rule G1)
kernel_A<<<gridA, blockA, 0, s1>>>();  // GPU op → s1 的 stream queue 尾部
kernel_B<<<gridB, blockB, 0, s1>>>();  // GPU op → s1 的 stream queue 尾部 (在 kernel_A 之后)

// GPU 端: Scheduler
while (stream_queues[s1] is not empty):
    head_op = stream_queues[s1].head
    if head_op is kernel:
        enqueue_EE_queue(head_op)  // Rule G2
        while not head_op.fully_dispatched:
            // Only head of EE queue is eligible (Rule X1)
            for each SM:
                if SM.available_threads >= head_op.threads_per_block
                   and SM.available_shmem >= head_op.shmem_per_block:
                    assign_block_to_SM(head_op, SM)  // Rules R1-R3
        dequeue_EE_queue(head_op)  // Rule G3
    if head_op is copy:
        enqueue_CE_queue(head_op)  // Rule C1
        assign_copy_to_CE(head_op)  // Rules C2-C3
        wait_copy_completion()
    dequeue_stream_queue(s1, head_op)  // Rule G4 / C4
```

术语一般如何实现？如何使用？
CUDA Stream 通过 CUDA Runtime API（cudaStreamCreate, cudaStreamDestroy）管理。CUDA 8.0+ 支持 stream priority (cudaStreamCreateWithPriority)。从 Kepler 架构引入 Hyper-Q 后，GPU 硬件层面支持多达 32 个并发工作队列，允许多 stream 真正并发。开发者通常使用多 stream 实现计算与数据传输 overlap（pipelining）或不同 kernel 的并发执行。注意事项：(1) 避免无意中使用 NULL stream 导致串行化；(2) 对于不需要 NULL stream 同步的 stream，使用 cudaStreamNonBlocking 标志创建。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---

## Thread Block (as Schedulable Entity)

术语是什么？
Thread Block（线程块）是一组可并发执行的 GPU 线程集合，是 GPU scheduler 进行调度决策的基本单位（schedulable entity）。一个 CUDA kernel 在 launch 时指定 block 数量和每个 block 的线程数。同一个 block 内的所有线程在同一 SM 上执行，可以通过 shared memory 和 __syncthreads() 进行协作。block 可以按任意顺序执行。

从kernel调度角度拆解术语：
在本文揭示的 TX2 GPU scheduler 中，"blocks are schedulable entities" 是核心发现。调度规则的关键：只有 EE queue 头部 kernel 的 block 才可被分配（Rule X1）；每个 block 的分配需满足目标 SM 的资源约束（Rules R1-R3: threads ≤ 2048/SM, shared memory ≤ 64KB/SM, registers ≤ 65536/SM）。不同 kernel 的 block 可在同一 SM 上同时执行（前提是都满足资源约束且符合 EE queue 优先级规则）。

具体计算过程——以论文 Fig. 3 的 K1 为例：
```
K1: 6 blocks × 768 threads/block, 0 shared memory/block
TX2 SM constraints: 2048 threads/SM, 64KB shmem/SM

Round 1 dispatch (Fig. 4(a)):
  SM0: 2 blocks of K1 assigned (2×768 = 1536 ≤ 2048 threads ✓)
  SM1: 2 blocks of K1 assigned (2×768 = 1536 ≤ 2048 threads ✓)
  Remaining: 2 blocks of K1 waiting (not enough thread resources)

Round 2 dispatch (Fig. 4(b)):
  After first 4 blocks complete, SM0 and SM1 each have 2048 threads freed
  SM0: 1 remaining block of K1 + 2 blocks of K4 assigned (768 + 2×256 = 1280 ≤ 2048 ✓)
  SM1: 1 remaining block of K1 + 2 blocks of K4 assigned (same)

K5 cannot be dispatched even though threads available:
  K5 requires 32KB shmem/block
  Each SM: 2 blocks of K4 × 32KB = 64KB = SM limit → no room for K5 (Rule R3)
```

术语一般如何实现？如何使用？
在 CUDA 编程中，kernel launch 的 `<<<gridDim, blockDim, sharedMem, stream>>>` 语法定义了 block 的配置。Block 大小的选择需要权衡 occupancy（SM 上同时驻留的 block/warp 数量）和资源使用（寄存器、shared memory）。在实时系统中，block 是 GPU 抢占的粒度边界——如本文发现 Pascal 架构支持指令级 preemption，但 block 仍是资源分配和调度的基本单位。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## NULL Stream (Default Stream)

术语是什么？
NULL Stream（也称为 Default Stream、Stream 0）是 CUDA 中隐式的默认流。在传统行为（legacy default stream）中，NULL stream 具有特殊的同步语义：当在 NULL stream 中执行操作时，先等待所有 blocking stream 完成，操作排队后，所有 blocking stream 再等待 NULL stream 完成。这导致 NULL stream 成为一个隐式全局同步屏障，可能严重破坏并发性能。

从kernel调度角度拆解术语：
论文通过实验发现了 NULL stream 在 TX2 GPU scheduler 中的具体调度规则（N1-N2，扩展自 Rule G2）：
- Rule N1: NULL stream queue 头部的 kernel Kk 入队 EE queue 的条件是——对于每个其他 stream queue，该 queue 为空或该 queue 头部的 kernel 在 Kk 之后 launch。
- Rule N2: 非 NULL stream queue 头部的 kernel Kk 入队 EE queue 的条件是——NULL stream queue 为空或 NULL stream queue 头部的 kernel 在 Kk 之后 launch。

具体例子（论文 Fig. 5, Table 2）：
```
假设三个 stream: S1(normal), S2(normal), NULL stream
Launch order: K1(S1, t=0) → K2(NULL, t=0.2) → K3(S2, t=0.2) → K4(S2, t=0.4) → K5(NULL, t=0.6) → K6(S3, t=0.8)

Rule N1 效果：
  K2(NULL stream head) 入 EE queue 条件: S1的head(K1)在K2之前launch → K2必须等K1完成后才能入EE queue
  → K2被K1阻塞，直到K1执行完。在此期间：
  Rule N2 效果：K3 虽在 S2 头部，但 NULL stream queue 非空且 head(K2)在K3之前launch → K3不能入EE queue
  → K3和K4都被K2阻塞（即使GPU有空闲资源）

结论：NULL stream 造成严重的不必要阻塞和GPU容量浪费。K6本可与K3/K4并发执行，但因NULL stream的同步语义而被全部串行化。
```

术语一般如何实现？如何使用？
NULL stream 是 CUDA 向后兼容的设计。在现代 CUDA 开发中，最佳实践是：(1) 使用显式 stream 替代 NULL stream；(2) 编译时使用 --default-stream per-thread 标志使每个 CPU 线程有独立的默认流；(3) 使用 cudaStreamNonBlocking 标志创建不与 NULL stream 同步的 stream。在实时系统中，NULL stream 的隐式同步行为应完全避免——论文明确指出"usage of the NULL stream is problematic if real-time predictability and efficient platform utilization are desired"。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## Stream Priority (CUDA)

术语是什么？
CUDA Stream Priority 是 CUDA 提供的流优先级机制。通过 cudaDeviceGetStreamPriorityRange 查询设备支持的优先级范围，通过 cudaStreamCreateWithPriority 创建带优先级的流。在 TX2 上，仅支持两个优先级值：-1（priority-high）和 0（priority-low）。未指定优先级的 stream（priority-none）被视为 priority-low。

从kernel调度角度拆解术语：
论文通过实验（Fig. 6-8）推导出 TX2 上 stream priority 的调度规则（A1-A2）：
- Rule A1: kernel 只能入队与其 stream 优先级匹配的 EE queue（priority-high queue 或 priority-low queue）
- Rule A2: EE queue 头部 kernel 的 block 可被分配，仅当所有更高优先级的 EE queue 为空

这意味着：
1. Priority-high stream 的 kernel 可以"抢占"priority-low stream 的 kernel——不是真正的中断正在执行的 block，而是当有新 block 可被分配时，优先从 priority-high EE queue 选 block。
2. Priority-high 可能导致 priority-low 饥饿（Fig. 6: K1 在 priority-low stream，8 blocks 只执行了 4 个就被 K2(priority-high) 抢占，剩余 4 个需等 K2 和 K3 全部完成后才能执行）。
3. 资源阻塞不影响优先级：若 priority-high kernel 因资源不足无法 dispatch，priority-low kernel 即使资源满足也不能"插队"（Fig. 8）。

伪代码——优先级调度逻辑：
```
function select_block_for_assignment(SM):
    if priority_high_EE_queue is not empty:   // Rule A2
        head_kernel = priority_high_EE_queue.head  // Rule A1
    else if priority_low_EE_queue is not empty:
        head_kernel = priority_low_EE_queue.head
    // Rule X1: only head kernel's blocks are eligible
    if SM has sufficient resources for head_kernel.block:
        assign block of head_kernel to SM
        // Note: blocks from non-head kernels are NOT eligible
```

术语一般如何实现？如何使用？
CUDA Stream Priority 适用于需要区分延迟敏感（priority-high）和吞吐优先（priority-low）工作负载的场景。注意：(1) TX2/Jetson 平台仅支持 2 个优先级级别；(2) 多 process 场景下 stream priority 无效（论文附录 A 发现）；(3) 高优先级可能导致低优先级饥饿，需要谨慎设计以避免无限期延迟。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## Hierarchical FIFO Scheduling (GPU)

术语是什么？
Hierarchical FIFO Scheduling（层次化 FIFO 调度）是本文发现并命名的 TX2 GPU scheduler 的调度策略。它是一种多级 FIFO 队列结构：多个 stream queue（每 stream 一个 FIFO）→ EE queue（per-address-space FIFO）→ SM assignment。每一级都使用 FIFO 顺序，但存在因资源约束和 stream 间依赖导致的 blocking delay，使得调度不完全是 work-conserving。

从kernel调度角度拆解术语：
调度的完整流程（论文 Rules G1-G4, X1, R1-R3, C1-C4, N1-N2, A1-A2）：

```
Level 1 — Stream Queues (per stream):
  In:  CUDA API calls (cudaLaunchKernel, cudaMemcpyAsync)
  Out: Kernel → EE queue (G2); Copy → CE queue (C1)
  Ordering: FIFO within each stream
  Blocking: Operations wait in stream queue until previous ops complete (G4/C4)

Level 2 — EE Queue (per address space):
  In:  Kernels from heads of stream queues (G2)
  Out: Block assignment to SMs (X1, R1-R3)
  Ordering: FIFO (launch-time order across streams)
  Non-preemptive: Only head kernel's blocks eligible (X1)

  With stream priority (A1-A2):
    Two EE queues: priority-high and priority-low
    High-priority queue must be empty before low-priority blocks assigned

Level 2b — CE Queue:
  In:  Copy ops from heads of stream queues (C1)
  Out: Assignment to CE (C2-C3)
  Ordering: FIFO
  Non-preemptive: Only head copy assigned at a time

Level 3 — SM Assignment:
  In:  Blocks from head of EE queue
  Constraints: threads ≤ 2048/SM, shmem ≤ 64KB/SM, registers ≤ 65536/SM
  Assignment: Greedy — assign eligible block to any SM with sufficient resources
```

论文指出这种调度"具有可分析的响应时间边界"——类似于多处理器上的 FIFO 调度已被证明具有可分析的 tardiness bounds [13]，因此 TX2 的 GPU scheduler 可能适用于实时可调度性分析。

术语一般如何实现？如何使用？
这种调度策略是 NVIDIA GPU 的硬件+驱动实现，对 CUDA 开发者透明。了解其行为对实时系统设计至关重要：(1) 使用多 stream 而非单 stream 可提高并发；(2) 避免 NULL stream 的隐式同步；(3) 资源约束导致的 blocking 不可避免，但可预测；(4) 在 task 共享地址空间场景下调度更可预测（vs process 独立地址空间场景）。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## GPU Multiprogramming / Compute Preemption (Pascal)

术语是什么？
GPU Multiprogramming（GPU 多道程序）指多个 CUDA 程序（来自不同 CPU process）的 kernel 在 GPU 上交替执行。在 Pascal 架构之前（如 TX1/Maxwell），多道程序通过 thread block 级别的轮转实现——不同 process 的 kernel 不能同时执行。Pascal 架构引入了 Compute Preemption（计算抢占），支持指令级抢占，允许 GPU 在执行中保存完整上下文（寄存器、shared memory、程序计数器等）到 GPU DRAM 并切换到另一 process 的 kernel。

从kernel调度角度拆解术语：
论文附录 A 揭示了 TX2（Pascal 架构）在多 process 场景下的调度行为：
- 不同 process 的 kernel 通过 preemption 实现 time-slicing 多路复用（而非真正并发）
- 现象：两个 process 各 launch 4096 线程的 kernel，GPU timeline 显示每个 kernel 看起来始终有 4096 线程在运行（但这超过了 TX2 的物理 4096 线程上限）——原因是 block 被抢占并恢复时，GPU 端记录的 start/end 时间戳实际包含了被抢占的时间段。
- 影响：多 process 场景下 block 执行时间可能翻倍（Fig. 10: Process 曲线的 worst-case block time 超过 Task 曲线的 2 倍）
- Priority 无效：stream priority 在多 process 场景下无效——只有 task 共享地址空间场景下 priority 才起作用。

```
Task 共享地址空间 (本文核心场景):
  Kernel K1 和 K2 可在不同 SM 上真正并发执行
  → 可预测的执行时间，FIFO 调度规则适用

Process 独立地址空间:
  Kernel K1 和 K2 通过 time-slicing 交替执行
  → block 被抢占导致执行时间膨胀，不可预测
  → 推荐使用 task 共享地址空间模型进行实时系统设计
```

术语一般如何实现？如何使用？
Compute Preemption 是 Pascal 架构引入的硬件特性，由 GPU driver 自动管理，不通过 CUDA API 暴露给程序员。它在以下场景被使用：(1) 多 process 的 time-slicing 调度；(2) 单 GPU 交互式 kernel 调试；(3) 防止长时间运行的 kernel 导致显示无响应（desktop GPU）。MPS (Multi-Process Service) 可以绕过 process 间 time-slicing，允许多 process 共享一个 CUDA context 实现真正并发。在 Jetson 嵌入式平台，推荐使用 task（共享地址空间）而非 process 来最大化 GPU 利用率和可预测性。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## Kernel Dispatch

术语是什么？
Kernel Dispatch 是 GPU scheduler 将 CUDA kernel 的 thread block 分配到 SM 上执行的过程。在本文的术语中，一个 kernel 被 dispatched 意味着至少一个 block 被分配到 SM；fully dispatched 意味着所有 block 已被分配。Block 是 dispatch 的基本单位，dispatch 受限于 SM 的资源约束（线程、shared memory、寄存器）和 EE queue 的状态。

从kernel调度角度拆解术语：
Dispatch 的具体流程（整合论文 Rules G2-G3, X1, R1-R3）：

```
函数: dispatch_kernel_from_EE_queue()
输入: EE_queue (FIFO)
输出: blocks assigned to SMs

while EE_queue not empty:
    head_kernel = EE_queue.head
    for each SM in GPU:
        if head_kernel has unassigned blocks:
            block = head_kernel.next_unassigned_block
            if SM.meets_resource_constraints(block):
                // Rules R1-R3
                SM.assign_block(block)
                mark block as assigned
            else:
                continue to next SM  // resource blocking
        else:
            break  // all blocks assigned
    if head_kernel.fully_dispatched:
        EE_queue.dequeue(head_kernel)  // Rule G3
    else:
        break  // head kernel blocked by resources => NO preemption of head (Rule X1)
               // Later kernels in EE queue cannot be dispatched even if resources exist
```

Dispatch 的关键特性：
1. Head-of-line blocking: EE queue 头部 kernel 因资源不足无法 fully dispatched 时，后续 kernel 即使资源满足也不能 dispatch（Rule X1: 非抢占）。
2. Resource blocking: 一个 block 只能分配到一个 SM（不能跨 SM 拆分）。例如，若 kernel 需要 1024 threads/block 且 SM0 有 512 空闲线程、SM1 有 512 空闲线程，该 block 无法分配（因为任何一个 SM 都不满足 1024 线程的要求）。
3. Stream FIFO 与 dispatch 的交互：kernel 在 stream queue 中阻塞（等待前面 kernel 完成）期间不会出现在 EE queue 中，因此不影响其他 stream 的 dispatch。

术语一般如何实现？如何使用？
Kernel dispatch 由 GPU 硬件和 driver 自动完成，对 CUDA 程序员透明。但理解 dispatch 行为对性能调优和实时系统分析至关重要：通过控制 block 的资源需求（thread 数、shared memory、register 数），可以预判 dispatch 的并行度和阻塞情况。CUDA Occupancy API 可以帮助开发者确定最优配置以最大化 SM 利用率。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## 8-WAVE PING-PONG Schedule (AMD)

术语是什么？
8-WAVE PING-PONG 是 HipKittens 论文提出的 AMD GPU kernel 调度的核心模式之一。它在一个 thread block 中使用 8 个 wave——每 SIMD 驻留 2 个 wave。8 个 wave 分为两组各 4 个（每组包含每个 SIMD 各 1 个 wave），通过 conditional barrier 交替执行 compute 和 memory 角色：组 A 的 wave 发射 MFMA 计算指令时，组 B 的 wave 发射 buffer_load_dword 从 HBM 预取下一 tile 到 LDS；完成后角色互换（ping-pong）。

从kernel调度角度拆解术语：
以 BF16 GEMM kernel (256x256 output tile per thread block, K_STEP=64) 的调度流程为例：

```
// Prologue: 8 waves 协作 preload
G::load(Bs[t0][0], g.b, {0, 0, col*2, 0});
G::load(As[t0][0], g.a, {0, 0, row*2, 0});
// 条件 barrier: 一半 wave 被 stall
if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }
// Leader wavegroup 继续 preload, 完成后释放 follower
// ...
__builtin_amdgcn_s_barrier();

// Hotloop: ping-pong 交替
for (tile = 0; tile < num_tiles - 2; ++tile) {
    // Leader compute cluster (follower 同时做 memory):
    load(B_tile_0, st_subtile_b);       // LDS to register
    load(A_tile, st_subtile_a);
    G::load(As[t_next][1], ...);        // async HBM to LDS
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_setprio(1);      // 提升 compute wave 优先级
    mma_ABt(C[0][0], A_tile, B_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    
    // 下一阶段: 角色交换 (Leader 做 memory, follower 做 compute)
    // 对称的 load+mma 操作
}
```

8-WAVE 允许使用大 tile 原语（类似 NVIDIA wave specialization），代码紧凑（GEMM hotloop 约 48 LOC），适合 compute 和 memory 持续时间平衡的 workload。在 MI355X 上，8-WAVE BF16 GEMM 达到 1610 TFLOPS，FP8 GEMM 达到 3222 TFLOPS，均匹敌 AITER 手写汇编。

术语一般如何实现？如何使用？
HipKittens 通过 C++ template 调度模板实现，开发者设置每 SIMD 2 个 wave，在 conditional stagger 后用 s_barrier 交替。适合 GEMM、attention forward 等 compute-memory 平衡 kernel。对 attention backwards 等 memory-heavy workload，4-WAVE INTERLEAVE 更优。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Wave Specialization (Producer-Consumer) on AMD

术语是什么？
Wave specialization（producer-consumer scheduling）是 GPU kernel 调度模式：将 wave/warp 分为专职 producer（仅做 HBM→shared memory 数据搬运）和 consumer（仅做 shared memory→compute）。在 NVIDIA GPU 上占主导（FlashAttention-3、CUTLASS、TK），通过 TMA、wgmma、mbarrier 等硬件特性实现高效 overlap。但在 AMD CDNA GPU 上，wave specialization 因架构差异而性能退化。

从kernel调度角度拆解术语：
在 AMD 上 wave specialization 的限制：
1. 静态寄存器分配：每 SIMD 512 寄存器在所有 wave 之间平均分配，producer wave 只需少量地址计算寄存器但仍占用大量 register，consumer wave 无法回收，缩小了 output tile size 和 arithmetic intensity。
2. 无 TMA/wgmma/mbarrier：缺少异步 HBM→shared memory 搬运（buffer_load 仍需 s_waitcnt 同步）、无 shared memory 直接矩阵乘、software barrier 开销大。
论文 Table 2 验证：随 producer 数量增加，性能下降（4P+8C: 893 TFLOPS，0P+8C: 1281 TFLOPS）。因此在 AMD 上推荐 8-WAVE PING-PONG（无专职 producer）或 4-WAVE INTERLEAVE。

术语一般如何实现？如何使用？
在 NVIDIA 上通过 TMA（cp_async_bulk）+ wgmma + mbarrier 实现。在 AMD 上，论文证明不使用 wave specialization（0 producer）配合 8-WAVE PING-PONG 性能最优。HK 通过模板参数控制调度模式。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Pinned Register Tiles (Explicit Register Scheduling on AMD)

术语是什么？
Pinned register tiles 是 HipKittens 的开发者显式控制 GPU 寄存器分配的机制。绕过 HIPCC 编译器的限制（如不允许 AGPR 作为 MFMA 输入操作数），直接指定每个 tile 映射到哪些物理寄存器（VGPR/AGPR 编号范围），实现 AGPR 直接作为 MFMA 的 A/B operand。在 attention backwards kernel 中，将性能从 855 TFLOPS 提升到 1024 TFLOPS（匹敌 AITER 汇编 1018 TFLOPS）。

从kernel调度角度拆解术语：
开发者通过寄存器范围定义 tile：
```
// 定义寄存器范围: v[24:27], v[28:31], v[32:35], v[36:39]
using Q_ranges = split_many_t<type_list<range<24, 39>>, 4>;
// 绑定到 tile
rt<bf16, 16, 128, row_l, rt_16x32_s, Q_ranges> Q_i;
```
API 与标准 compiler-managed tile 完全一致（load/mma/store），开发者可选择控制粒度。FP6 GEMM 中，explicit register scheduling 完全消除了 54-register scratch spill。

术语一般如何实现？如何使用？
C++ template 元编程实现，ranges 参数映射到具体寄存器编号。仅在编译器限制导致性能损失时使用（attention backward、FP6 等），正常 kernel 仍用 compiler-managed tile。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## MFMA (Matrix Fused Multiply Add) — AMD Matrix Core Instruction

术语是什么？
MFMA 是 AMD CDNA GPU 的矩阵核心指令，执行 D=A*B+C。由 wave 的 64 线程协作完成。AMD 使用较小形状（16x16x32、32x32x16、16x16x128）vs NVIDIA 的 256x256x16，且各形状使用完全不同的 thread-to-element mapping（无 NVIDIA 的统一 16x16 core matrix 结构）。CDNA4 新增 FP6/FP4 scaled MFMA。

从kernel调度角度拆解术语：
HipKittens BF16 GEMM 中，每个 warp 使用 16x16x32 MFMA 计算 128x64 输出子 tile：
```
mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
// 底层: v_mfma_f32_16x16x32_bf16 指令, 16 cycle 发射延迟
```
HipKittens 默认用最小 MFMA 形状（16x16x32）以最大化调度灵活性 + deep pipeline，与 NVIDIA 偏好大 MFMA 形状（利用 wgmma 从 shared memory 直接矩阵乘）相反。

术语一般如何实现？如何使用？
通过 LLVM builtin（__builtin_amdgcn_mfma_*）在 HIP kernel 中使用。HipKittens 的 mma_ABt 等自动选择正确指令变体。AMD Matrix Instruction Calculator（https://github.com/ROCm/amd_matrix_instruction_calculator）可查询各形状的寄存器布局。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## AGPR / VGPR (AMD GPU Register Types)

术语是什么？
AGPR（Accumulator GPR）和 VGPR（Vector GPR）是 AMD CDNA SIMD 的两种寄存器。每 SIMD 512 个 32-bit 寄存器，单 wave 时分为 256 VGPR + 256 AGPR。VGPR 用于向量运算和访存，AGPR 用于 MFMA 累加器。HIPCC 不允许 AGPR 作为 MFMA 输入，即使硬件支持，导致编译器插入冗余 v_accvgpr_read 指令。HipKittens 通过 pinned register tiles 绕过此限制。

从kernel调度角度拆解术语：
在 attention backwards 中，MFMA 累加结果在 AGPR，后续需做 softmax vector 运算，HIPCC 必须插入 v_accvgpr_read AGPR→VGPR 搬移。Pinned tiles 直接指定 AGPR 范围作为 MFMA 输入，消除搬移开销。LLVM Dec 2025 patch (PR #170335) 开始自动支持 VGPR→AGPR rewrite。

术语一般如何实现？如何使用：
由硬件和编译器管理，__launch_bounds__ 限制每 wave 寄存器数。HipKittens pinned tiles 在需要精确控制时使用。CDNA4 新增的 scaled MFMA 支持 FP6/FP4 时，AGPR 管理更加关键。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Chiplet Swizzling (XCD Swizzle — Cache-Aware Grid Scheduling)

术语是什么？
Chiplet swizzling 是 HipKittens Algorithm 1：在 AMD MI355X 8-XCD chiplet GPU 上，通过 remap thread block 的 grid 坐标，优化两级缓存（每 XCD 私有 4MB L2 + 全局 LLC）的数据复用。默认 row-major grid 下 L2 hit rate 仅 36%-55%；Algorithm 1 通过 XCD grouping（chunks of C blocks 归同一 XCD）和 hierarchical windowed traversal（W 高度的垂直窗口遍历），将 L2 hit rate 提升至 78-79%，整体带宽提升 19%。

从kernel调度角度拆解术语：
```
Algorithm 1 流程:
1. Flatten (b.x, b.y) to linear ID
2. XCD grouping: 连续 C blocks → 同一 XCD
3. Hierarchical windowing: W 行 × num_cols 列的窗口内分配
4. 窗口内优先沿列方向（同一 XCD 覆盖矩形 L2 tile）
5. 跨 XCD 窗口对齐（重叠 A 行和 B 列 → 提升 LLC hit rate）
6. 尾部 block 保持原始顺序
```
参数 W（窗口高度，控制 L2 reuse）和 C（chunk 大小，控制 LLC reuse）的权衡：L2 带宽约 3x LLC 带宽，优先最大化 L2 hit rate。

术语一般如何实现？如何使用？
在 GEMM kernel launch 前 CPU 端执行，将 remap 后的 (row, col) 传给 kernel。当输出 tile 数与 XCD 数互质时收益最大。可推广到其他 workload。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Tile-Based GPU Kernel Programming (ThunderKittens / HipKittens)

术语是什么？
Tile-based programming 以 tile（二维数据块）为基本数据结构，提供 PyTorch/NumPy 风格的 bulk 操作符注册 tile 上。ThunderKittens (NVIDIA) 和 HipKittens (AMD) 通过 C++ template 元编程实现，内部直接包装 PTX/CDNA assembly。核心抽象：register tile (rt_bf/rt_fl)、shared memory tile (st_bf)、load/store operators 和 compute operators (mma、exp、add 等)。

从kernel调度角度拆解术语：
HipKittens tile 接口示例（BF16 GEMM）：
```
// shared memory tiles (double buffered)
st_bf<128, 64, st_16x32_s> As[2][2], Bs[2][2];
// register tiles
rt_bf<64, 64, row_l, rt_16x32_s> A_tile, B_tile;
// accumulator
rt_fl<64, 128, col_l, rt_16x16_s> C_accum[2][2];

G::load(Bs[t][0], g.b, ...);         // HBM→LDS
load(B_tile, subtile(Bs[t][0], ..)); // LDS→register
mma_ABt(C[0][0], A_tile, B_tile, C[0][0]); // compute
store(g.c, C[0][0], ...);            // register→HBM
```
框架自动处理 AMD 特有复杂性：异构 MFMA layout、phase/bank behavior、buffer_load swizzle 地址计算。tile 抽象已被验证可从 NVIDIA 移植到 AMD，表明统一的 tile-based 编程模型可能成为跨厂商通用 kernel 开发范式。

术语一般如何实现？如何使用？
HipKittens 是 C++ header-only 库（https://github.com/HazyResearch/HipKittens），`#include` 使用。ThunderKittens (https://github.com/HazyResearch/ThunderKittens) 用于 NVIDIA。Python bindings 通过 pybind11 集成到 PyTorch。tile 的 row/col 尺寸必须为 MFMA 形状的整数倍。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels
- ThunderKittens: Simple, Fast, and Adorable Kernels

---

## buffer_load_dword (AMD Asynchronous Global-to-LDS Load)

术语是什么？
buffer_load_dword 是 AMD CDNA3/4 的异步 HBM→LDS（shared memory）加载指令，等价于 NVIDIA TMA。数据绕过 register file 直接写入 LDS，wave 可继续执行其他操作。变体：dword (4B)、dwordx3 (12B)、dwordx4 (16B)。通过 s_waitcnt vmcnt(N) 等待未完成的 load。

从kernel调度角度拆解术语：
在 HipKittens GEMM hotloop 中：
```
G::load(Bs[t][0], g.b, {0, 0, col*2, tile+2}); // 异步发射
// ... 执行 MFMA 和其他操作 ...
asm volatile("s_waitcnt vmcnt(6)");  // 等待直到 <=6 个 vm 操作未完成
__builtin_amdgcn_s_barrier();
```
全局地址的 swizzle 在 HBM 地址阶段完成（与 NVIDIA TMA 在 shared memory 地址上 swizzle 不同）。buffer_load_dwordx4 每个 thread 加载 16 bytes，最小化指令发射数但可能引起 shared memory alignment 问题（如 FP6 kernel 需 16-byte aligned ds_read_b128）。

术语一般如何实现？如何使用：
通过 HIP 内嵌汇编使用。截至 Sep 2025，Triton AMD 仍未默认使用 buffer_load（需 PR #8013 手动启用）。HipKittens 的 G::load 模板封装了指令选择、地址计算和 swizzle。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Shared Memory Bank Conflict / Swizzle on AMD CDNA

术语是什么？
AMD CDNA shared memory bank conflict 发生在 wave 内多线程同一 phase 访问同一 bank。Swizzle 通过重排数据布局消除冲突。AMD banking 比 NVIDIA 更复杂：不同指令使用不同 bank 数量（64/32）和不同 phase ordering（非顺序的线程分组），且 AMD MFMA 各形状使用完全不同的 thread-to-element mapping，单一 swizzle 无法覆盖所有访问模式。

从kernel调度角度拆解术语：
HipKittens 自研 solver 反向工程 phase/bank 行为（Table 5），识别常共同出现的 layout 对（如 16x32 row+col load），为这些对设计 bank-conflict-free swizzle。例如 ds_read_b128 要求 128-bit 连续，但 ds_write_b64 的 XOR swizzle 以 64-bit 为单位打散数据，两者冲突。在 attention backward 中需两个不同 swizzle 分别服务于 row-layout 16x16 write（ds_write_b64）和 row-layout 16x32 read（ds_read_b128）。

术语一般如何实现？如何使用？
AMD phase/bank 行为未在 ISA 手册中文档化。HipKittens solver 自动探测。Swizzle 通过 XOR/移位在全局地址（HBM offset）或 LDS offset 上实现。CDNA4 新增指令（如 ds_read_b96）有新的 phase 行为。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---

## Data Flow Graph (DFG) for GPU Kernel Scheduling

术语是什么？
Data Flow Graph (DFG) 是一种有向图结构，用于表示GPU程序中各kernel之间的数据依赖关系。在HuntKTm的stream scheduler中，DFG的每个节点是一个kernel调用，有向边表示数据依赖（RAW/WAW/WAR）。DFG的构建是自动kernel并行化的基础——通过DFG可识别哪些kernel没有相互依赖，可以分配到不同CUDA stream上并发执行。DFG还支持分层（levelization）：同一层内的kernel互相无依赖，可以安全并发。

从kernel调度角度拆解术语：
HuntKTm的DFG constructor通过以下流程自动构建DFG：
1. 开发者仅在每个kernel参数列表开头插入一个常量，标注writable参数个数N_out，并将writable参数重排到前N_out位置。
2. DFG constructor逆序遍历kernel调用序列，使用BFS算法识别每个kernel的直接前驱（predecessor）。
3. 依赖判断规则：若两个kernel访问同一数据对象，且至少一个访问是写操作，则存在依赖。具体而言：
   - kernel A（在B之前执行）写data，kernel B读data → RAW依赖 A→B
   - kernel A读data，kernel B写data → WAR依赖 A→B
   - kernel A写data，kernel B写data → WAW依赖 A→B
   - 两者均只读data → 无依赖
4. 指针别名处理：通过同一base address派生的指针参数被视为访问同一数据，确保指针算术不会导致遗漏依赖。

```
构建DFG伪代码:
Function buildDFG(kernelList):
    dfg = empty DAG
    for i = kernelList.length-1 down to 0:  // 逆序遍历
        kernel_i = kernelList[i]
        for j = i-1 down to 0:  // BFS搜索前驱
            kernel_j = kernelList[j]
            for each data object d:
                if kernel_i.writes(d) OR kernel_j.writes(d):
                    if kernel_i.accesses(d) AND kernel_j.accesses(d):
                        dfg.addEdge(kernel_j → kernel_i)
                        break  // 找到直接前驱后停止
    return dfg
```

DFG分层（Levelization）：
```
level = {0, 0, ..., 0}
for each node in topological_order:
    for each predecessor p of node:
        level[node] = max(level[node], level[p] + 1)
// 同level的kernel无相互依赖，可分配到不同stream并发
```

术语一般如何实现？如何使用？
HuntKTm通过LLVM pass在host IR中定位`__cudaPushCallConfiguration`调用模式来识别kernel launch，对识别到的kernel执行DFG构建。相比GrSched的运行时动态DFG构建，编译期DFG构建可以获得全局视图，支持更激进的同步剪除优化。对复杂控制流（如kernel调用在循环或条件分支中），HuntKTm退化为保守的依赖分析。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---

## Preferred Predecessor Set (PP-Set)

术语是什么？
Preferred Predecessor Set (PP-Set) 是HuntKTm的kernel distributor中使用的启发式概念。对于一个尚未被调度的kernel，其PP-Set定义为：该kernel的所有前驱（predecessor）中，当前位于各自所在stream末尾（即stream中的最后一个已调度kernel）的那些前驱组成的子集。PP-Set的大小直接影响该kernel的调度优先级——PP-Set越小的kernel越先被调度，以最小化跨stream同步的数量。

从kernel调度角度拆解术语：
PP-Set在kernel分配算法中的作用——以论文Figure 6的DFG为例：

```
初始: Stream1=[], Stream2=[], Stream3=[]
Level 1 (kernel A, B, C无前驱):
  Rule ∂ (round-robin): A→S1, B→S2, C→S3
  所有kernel PP-Set=∅

Level 2 (D依赖B; E依赖A,C; F依赖C):
  第1轮排序:
    F: PP-Set={C}, size=1
    D: PP-Set={B}, size=1
    E: PP-Set={A, C}, size=2
    → 先调度F (Rule ∑, 单前驱C → 同Stream3)
  
  PP-Set更新: C不再是Stream3末尾(F现在是末尾)
    E: PP-Set={A}, size=1
    D: PP-Set={B}, size=1
    → 调度D (Rule ∑ → D放入Stream2, 同B)
  
  最后调度E: PP-Set={A} → Rule ∑ → E放入Stream1 (同A)

最终: S1=[A,E], S2=[B,D], S3=[C,F]
跨stream同步: 仅需E→C的barrier (D和F无额外同步)
```

术语一般如何实现？如何使用？
PP-Set是HuntKTm kernel distributor内部的启发式数据结构，在每次kernel调度后动态更新。核心洞察：将kernel放在其PP-Set中某个前驱所在的stream中，可以避免为该前驱创建跨stream同步（因为同stream内的串行执行已隐式保证顺序）。优先调度PP-Set小的kernel，给予它们更多灵活的stream选择空间。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---

## Synchronization Pruning (GPU Stream Barriers)

术语是什么？
Synchronization Pruning（同步剪除）是HuntKTm的synchronization generator中的优化技术。当kernel被分配到多个CUDA stream后，必须在有数据依赖的跨stream kernel之间插入CUDA event同步（barrier）。但并非所有依赖都需要显式同步——许多barrier因为依赖传递性（transitivity）和同stream内隐式串行执行而冗余。Synchronization pruning通过三步算法识别并移除这些冗余barrier，只保留最小必要的同步集。

从kernel调度角度拆解术语：
HuntKTm同步剪除算法的三步（以kernel K为目标）：

```
Step 1 (创建初始barrier):
  for each predecessor P of K:
    if P.stream != K.stream:
      createBarrier(P → K)

Step 2 (每stream保留最后一个前驱):
  for each stream S:
    predecessors_in_S = K在S中的所有前驱
    if predecessors_in_S非空:
      仅保留最后执行的前驱的barrier, 移除其余
      // 理由: 同stream FIFO保证更早的前驱一定先于最后前驱完成

Step 3 (隐式同步传递性剪除):
  for each kernel T in K.stream (T在K之前):
    for each predecessor P of K:
      if P在T的stream中 AND P在T之前执行:
        removeBarrier(P → K)
        // K等待T(同stream) AND T等待P(同stream) → K隐式等待P
```

论文Figure 6(b)展示了剪除效果：实线为保留barrier，虚线为被剪除barrier。相比naive为每对依赖创建barrier，剪除后barrier数量显著减少。

术语一般如何实现？如何使用？
编译期LLVM pass实现，具有全局DFG视图——这是相比GrSched（运行时无全局视图）的关键优势。剪除后的CUDA event同步通过`cudaEventRecord`和`cudaStreamWaitEvent` API插入。正确性保证：仅移除那些执行顺序已由其他同步路径隐式保证的barrier。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---

## Memory Liveness Analysis for GPU Multi-Stream Programs

术语是什么？
Memory Liveness Analysis（内存活跃度分析）是针对GPU多stream程序的内存对象生命周期分析技术。在HuntKTm的memory manager中，通过数据流分析确定每个GPU memory object的live range（从首次被kernel访问到末次被kernel访问的区间），然后将allocation/deallocation指令调度到live range边界处，缩短lifetime至live range。非重叠live range的memory object可复用同一块GPU内存，从而降低peak memory usage。

从kernel调度角度拆解术语：
HuntKTm memory liveness analysis流程（Algorithm 2）：

```
Step 1 - 数据流分析:
  for each kernel call:
    for each GPU pointer parameter:
      通过use-def chain追踪pointer指向的allocation指令
      → 同一allocation分配的所有内存视为一个memory object
      → 所有使用该object的kernel标记为依赖

Step 2 - Live range识别:
  invokeList = 所有依赖memObj的kernel (按原始顺序排序)
  liveRange = [invokeList[0], invokeList[last]]

Step 3 - 延迟Allocation (PostponeMalloc):
  instrList = memObj的allocation + 内容修改指令(cudaMemcpy等)
  insertPoint = invokeList[0]  // live range起点
  将instrList移到insertPoint之前, 转为异步版本
  分配到与insertPoint相同的stream
  添加跨kernel同步确保memObj在使用前已分配

Step 4 - 提前Free (对称算法):
  将cudaFree移到live range终点kernel之后

Step 5 - 冗余同步剪除
```

论文结果：M2从17.6GB降至11.2GB（-36.4%），DL从7.06GB降至4.70GB（-33.3%），平均memory reduction 22.3%。

术语一般如何实现？如何使用？
LLVM pass编译期实现。限制：当前仅处理每memory object最多一次host-device数据传输；多stream可能缩短执行路径减少memory reuse机会（如B&S的10个kernel分配到10个stream后同stream仅1个kernel，无reuse可能）。Peak memory runtime预测使用O(N)近似算法，取各stream最大累积delta memory之和作为上界。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---

## Cross-Operation Kernel Fusion (in LLM-Generated Kernels)

术语是什么？
Cross-Operation Kernel Fusion是将多个逻辑上分离的算子融合为单个Triton kernel launch的优化技术，消除intermediate tensor materialization (HBM round-trip)和kernel launch overhead。KernelEvolve通过LLM agent的graph-based search自动发现和实现fusion——agent通过系统化搜索explore不同operator compositions和tiling configurations，profiling feedback automatic ranking fusion策略的有效性。

从kernel调度角度拆解术语：
论文中三个关键fusion案例：

**Conv1d Fusion** (5 kernels → 2 kernels on H100):
```
PyTorch Conv1d:
  nchwToNhwcKernel(输入) → nchwToNhwcKernel(权重) → sm90_xmma_fprop_implicit_gemm
  → nhwcToNchwKernel → triton_poi_fused_convolution(后处理)
  5次launch, 4次layout transform + 1次compute

KernelEvolve Triton Conv1d:
  pack_conv1d_weight_kernel → conv1d_gemm_kernel
  2次launch, 无layout transform, 直接处理native 1D layout
  2.30× speedup vs conv1d, 1.62× vs conv2d workaround
```

**Optimized FM Fusion** (2 bmm → 1 kernel on H100):
```
PyTorch (torch.compile):
  独立bmm(1) → write intermediate X^TY to HBM → 独立bmm(2) → write output
  2 loads + 2 writes + 1 intermediate HBM round-trip

KernelEvolve Fused:
  Load X,Y tiles once → compute X^TY in SRAM → compute X·(X^TY) in SRAM → write output
  1 load + 1 write, intermediate stays in SRAM
  ~2× memory traffic reduction, 2-4× speedup (N ≤ 64)
```

**PFFN Fusion** (2 kernels, 3 passes → 1 kernel, 1 pass):
```
PyTorch: extern_kernels.bmm(pass1: load→bmm→write)
        → triton_per_fused_rms_norm_add_gelu(pass2: load+bias+RMSNorm stats)
        → (pass3: load+normalize apply)
KernelEvolve: single-pass: load→bmm+bias+GELU+RMSNorm+bmm+bias+RMSNorm→write
1.2-2.6× speedup on production shapes
```

术语一般如何实现？如何使用？
Agent通过搜索自动发现fusion opportunities，无需预知哪些operators可融合。关键约束是SRAM容量——fused kernel的所有intermediate results必须fit in SRAM；超出容量时fallback到unfused baseline。Deployment使用shape-specific dispatch：generated kernel用于production shapes（保证性能），fallback到PyTorch/vendor library用于out-of-distribution inputs（防止regression）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## Shape-Specific Tiling (Kernel Generation)

术语是什么？
Shape-Specific Tiling是KernelEvolve自动生成的kernel优化策略：针对production deployment的特定tensor shape distribution定制tile尺寸和layout，最大化SRAM utilization和data reuse。与compiler的generic autotuning（标准化tile sizes和heuristics）和手动开发（expert基于经验选择tile尺寸）不同，agent在生成阶段就incorporate production input ranges并通过search自动发现最优tiling。

从kernel调度角度拆解术语：
PFFN kernel的tiling行为分析（Figure 15）:
```
Production config: B=1024, N∈[150,400], D∈[96,256], K∈[96,256]

Speedup vs D (fixed B=1024, K=256):
  D≤100: 1.6-1.9×  → tile comfortably fits SRAM → effective fusion
  D≈200: 1.1-1.2×  → tile near SRAM limit → partial spilling
  D>200: 1.2-1.4×  → adaptive strategy → recovery via tile resizing
```

Non-monotonic behavior源于tile size和SRAM capacity的复杂交互——human expert需要extensive trial-and-error找optimal tiling，agent通过系统化搜索自动发现。

Conv1d kernel的specialization trade-off：production shape (2048,96,96,200)上2.30× speedup，但out-of-distribution shapes (64×768×768×1024)上仅0.49-0.63×——specialization以generality为代价。Deployment用shape-aware dispatch：target shapes用generated kernel，其他用vendor library fallback。

术语一般如何实现？如何使用？
KernelEvolve通过search每步evaluate tile configurations on production shapes（via get_inputs()），fitness feedback automatic ranking tiling策略。Expanded autotuning探索20+ configurations (BLOCK_M/N/K + num_warps + num_stages + pipeline stages)，keyed to input dimensions (key=["N"])在shape变化时re-autotune。Cross-operation tile reuse进一步利用SRAM——同一tile的loaded data完成整个operator chain后才写回HBM。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## Batched Preprocessing Kernels (Jagged Tensor Operations for Recommendation)

术语是什么？
Batched Preprocessing Kernels是处理推荐系统data preprocessing pipeline中irregular/jagged tensor操作的GPU/Accelerator kernel。与regular dense tensor GEMM不同，这些kernels处理：(1) jagged tensors (variable-length per sample)；(2) data-dependent control flow (binary search, conditional matching)；(3) nested indexing (batch→user→event→feature)。KernelEvolve通过自动生成fused Triton kernel解决emerging accelerators上native operator coverage不足的问题。

从kernel调度角度拆解术语：
**MapIdTransform** (Fused on MTIA):
```
PyTorch: bucketize → clamp → gather → where (4独立ops, MTIA v2i上部分op缺失需CPU fallback)

KernelEvolve Triton (single launch, MTIA-optimized):
  for block in parallel:
    values_tile = tl.load(coalesced, mask)
    // Compile-time loop unrolling: for _ in range(20) — supports up to 2^20 mappings
    left=0, right=|M|
    for _ in range(20):
        mid = (left+right)>>1
        left = tl.where(search_active & (values_tile > M[mid]), mid+1, left)
        right = tl.where(search_active & (values_tile <= M[mid]), mid, right)
    output = tl.where(M[left] == values_tile, left+1, 0)  // in-register match
    tl.store(coalesced, output, mask)
  
MTIA v2i: 3.28-4.07× speedup; v3: 1.05-1.36× (stronger baseline)
```

**MBDT** (SIMD-vectorized on MTIA):
```
PyTorch: per-feature, per-element torch.bucketize

KernelEvolve: SIMD-vectorized counting replaces binary search
  for border_val in borders:
      count += (values > border_val).to(int)  // 64-256 elements simultaneously
  // For 3-10 element border arrays: O(n) > O(log n) due to branch-free + no CF overhead
  
MTIA v2i: 2.94-9.25×; v3: 2.31-3.09×
```

**Batch Event Truncate** (Multi-feature parallel):
```
PyTorch: per-feature sequential loop (no batched variant existed)

KernelEvolve batched Triton: all features in parallel, single launch
  No truncation needed: 9.8-14.5× (single vectorized compare vs per-element loop)
  Truncation required: 1.4-2.0× (constant launch vs sequential iteration)
```

术语一般如何实现？如何使用？
KernelEvolve通过graph-based search自动生成fused preprocessing kernels。对MTIA v2i（native operator coverage不足），生成的kernels不仅是性能优化，更是functional enablement——唯一可行的on-device执行路径。对v3（coverage更完整），kernel fusion和hardware-specific tuning仍提供2-3× speedup。关键MTIA-specific优化：compile-time loop unrolling (for branchless binary search)、coalesced block-parallel execution、register-resident computation (no intermediate tensor allocation)、avoiding tl.where in loops (direct boolean-to-int conversion)。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---

## Synchronous Dataflow Execution on GPUs

术语是什么？
Synchronous Dataflow Execution 是一种 GPU 执行模型，将 DL 计算图中的不同算子映射到不同 CTA，通过片上队列（on-chip queue）传递 tile 级中间数据，使多个算子在不同 SM 上 **空间并发**（spatial concurrency）执行，而非 BSP 的串行执行或 Vertical Fusion 的时间复用（temporal multiplexing）。在 Kitsune 的形式定义中，属于 synchronous dataflow（SDF）：当数据到达输入队列时，CTA 开始执行；完成计算后将结果写入生产者队列以触发后继 CTA。子图的首节点从主存读取激活，末节点将结果写回主存。

从kernel调度角度拆解术语：
Kitsune 的 synchronous dataflow 执行通过两个原语实现：(1) 软件 ring buffer queue（L2-resident, atomics-based）实现 inter-CTA 通信；(2) modified grid scheduler（双 arbiter：SIMT + Tensor）实现异构 CTA 在 SM 上 colocate。

以 MLP spatial pipeline（Linear → ReLU → Linear）的 dataflow 执行为例：

```
// Host: 配置 spatial pipeline
cudaPipeline pipeline;
pipeline.addKernel(kernel_Linear1, CTA_count=64, type=TENSOR);
pipeline.addKernel(kernel_ReLU,   CTA_count=44, type=SIMT);
pipeline.addKernel(kernel_Linear2, CTA_count=44, type=TENSOR);
pipeline.addQueue(queue0, producer=Linear1, consumer=ReLU);
pipeline.addQueue(queue1, producer=ReLU, consumer=Linear2);
pipeline.launch();

// GPU 端执行:
// SM_0: Linear1_CTA_0 (TensorCore: 执行GEMM tiles)
//       ReLU_CTA_0   (SIMT Core:  执行elementwise) ← 双arbiter确保co-location
// SM_1: Linear1_CTA_1 + ReLU_CTA_1
// ...
// 不同stage CTA并发: dataflow触发执行
//   Linear1写tile→queue0→ReLU消费, ReLU写result→queue1→Linear2消费
//   全程无global barrier, 无DRAM round-trip
```

与 BSP 的关键区别：(a) No global barrier between operators；(b) intermediate data 通过 L2-resident queue 传递（非 DRAM round-trip）；(c) 多类型 CTAs 并发执行，充分利用 TensorCore 和 SIMT Core。

术语一般如何实现？如何使用？
Kitsune 通过 PyTorch Dynamo compiler backend 自动将 DL 图 lowering 到 dataflow pipeline：(a) Subgraph Selection → 模式匹配识别 sf-node；(b) Pipeline Design → 插入 queue 节点；(c) Load Balance → ILP 求解 CTA 分配；(d) Code Generation → CUDA kernel 改写为读写 queue。cudaPipeline API 指定 kernel type（SIMT/TENSOR），modified grid scheduler 用双 arbiter 配对不同类型 CTA 到同一 SM。Queue 性能：54 queues × 2 CTAs = 108 CTAs 对应 A100 108 SMs，aggregate bandwidth 2 TB/s（37 GB/s/queue @ 128-256KB payload），同步 overhead <63% @ ≥64KB payload。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---

## Bulk-Synchronous Programming (BSP) on GPUs

术语是什么？
Bulk-Synchronous Programming (BSP) 是 GPU 默认的执行模型：每个 DL 算子映射为单个 CUDA kernel，一个 kernel 独占 GPU，所有 CTA 完成后 global barrier，再 launch 下一个 kernel。源自 Leslie Valiant 的 BSP 并行计算模型（1990），GPU 采用简化版：每个 kernel 的 CTA 独立执行到完成 → implicit global barrier → 下一个 kernel。现代 GPU 虽通过 CUDA Streams 支持有限的多 kernel 并发，但 grid scheduler 的 FIFO 设计使 kernel 间几乎没有执行重叠——新 kernel 需等当前 kernel 全部 CTA dispatch 后才开始 dispatch。

从kernel调度角度拆解术语：
BSP 模型在 GPU 上的执行流程：

```
// PyTorch eager 执行 MLP: Linear→ReLU→Linear
kernel_Linear1<<<grid, block>>>(input, intermediate1);
// implicit barrier: GPU等待所有CTA完成
// intermediate1 写入DRAM (non-resident on-chip)

kernel_ReLU<<<grid, block>>>(intermediate1, intermediate2);
// barrier: 等待所有CTA完成
// intermediate2 写入DRAM

kernel_Linear2<<<grid, block>>>(intermediate2, output);
// barrier: 等待所有CTA完成
```

BSP 的三大缺陷（Kitsune 论文分析）：
1. **资源闲置**：单 kernel 执行时 TensorCore 或 SIMT core 之一空闲。论文实测 inference 中 20-25% runtime、training 中 37-67% runtime 中 SM 和 DRAM 利用率均 <33% 峰值。
2. **大 intermediate 溢写 DRAM**：MLP hidden dim ≥ 768（A100 192KB shared memory 约束）时 intermediate tile 超出 SM 片上容量，必须 round-trip DRAM（A100 latency ≈ 409ns/572 cycles）。
3. **无法利用 reduction/hidden 维度并行**：如 back-propagation 中 batch 维度 gradient reduction，仅少数 CTA 执行 reduce，大多数 SM 空闲。

Kitsune 通过 Synchronous Dataflow Execution 解决以上三个问题：(1) heterogeneous CTA co-location 解决资源闲置；(2) L2-resident queue 消除 DRAM spill；(3) parallel reduction tree via queue 解决 reduction 并行不足。

术语一般如何实现？如何使用？
BSP 是 GPU 的默认执行模型，由 CUDA driver + grid scheduler 硬件强制执行。开发者通过 CUDA kernel launch 使用，无需额外编程。垂直融合（Vertical Fusion）和 Kitsune dataflow 是在 BSP 上的不同突破：Vertical Fusion 在单个 CTA 内 temporal multiplex 多个算子避免 barrier；Kitsune 改用 spatial pipeline 实现真正的并发。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---

## Spatial Pipeline (on GPUs)

术语是什么？
Spatial Pipeline 是 Kitsune 提出的 GPU kernel 执行抽象：将 DL 计算图的不同算子映射为 pipeline 的不同 stage，各 stage 对应一组 CTA，stage 间通过片上 queue 传递 tile 级数据。不同 stage 的 CTA **在空间上并发**（不同 SM 或同一 SM 的不同执行单元），实现 operator 级的流水线并行。通过 cudaPipeline API 暴露给开发者，语义类似 CUDA Graphs 但要求所有 kernel co-resident on GPU。

从kernel调度角度拆解术语：
Spatial pipeline 的调度结构：

```
sf-node: {stage_0, stage_1, ..., stage_n}
Queue: {queue_0, queue_1, ..., queue_{n-1}}  // stage_i → queue_i → stage_{i+1}

ILP最优CTA分配 (Algorithm 2):
  maximize Throughput
  subject to:
    Throughput < t_i × ResourceScale(a_i) × Speedup(a_i)  // 每stage性能约束
    Throughput × DRAM_Bytes < DRAM_Bandwidth               // 内存带宽约束
    Throughput × L2_Bytes < L2_Bandwidth                   // L2带宽约束
    1 ≤ a_i ≤ #SMs
    Σ IsSimt_i × a_i = #SMs     // SIMT和Tensor类CTA重叠SM分配
    Σ IsTensor_i × a_i = #SMs   // 利用不同execution unit的独立性
```

以 MeshGraphNets MLP forward pass 为例：
```
sf-node = {Linear_1 (256×1024 GEMM), ReLU, Linear_2 (1024×256 GEMM)}
Queue_0: Linear_1 → ReLU (payload: 64-256KB tiles)
Queue_1: ReLU → Linear_2

ILP求解: a_Linear1=64, a_ReLU=44, a_Linear2=44
  → 64+44=108 SM for stage 0+1 (SIMT/Tensor overlap: ReLU用SIMT, Linear1用Tensor)
  → 44 SM for stage 2 (Linear2用Tensor, 可能与其他stage重叠)
  → 152 CTAs 压缩到108 SM预算内（通过类型互补）
```

与垂直融合的关键对比：(a) Spatial pipeline 中不同 stage 分布在**不同 CTA**（空间并行），垂直融合中不同 operator 在**同一 CTA**内 temporal multiplex（时间复用）；(b) Spatial pipeline 通过 queue 在 CTA 间传递数据（L2 resident），垂直融合通过 shared memory 在 CTA 内部传递；(c) Spatial pipeline 支持隐藏维度/归约维度的并行（通过 queue 的多对一拓扑），垂直融合不支持。

术语一般如何实现？如何使用？
Kitsune compiler 自动 lowering：Subgraph Selection → Pipeline Design → Load Balance (ILP) → CUDA kernel 改写（每个 kernel 约 8 人时手动改写，10-40 LOC）。Modified grid scheduler 的 cudaPipeline API 带 kernel type metadata（SIMT/TENSOR），双 arbiter 实现异构 CTA pairing。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---

## Inter-CTA Ring Buffer Queue (L2-Resident)

术语是什么？
Inter-CTA Ring Buffer Queue 是 Kitsune 的纯软件实现的 CTA 间通信队列。使用 L2 cache 作为存储介质 + global atomics 作为同步机制，在 producer CTA 和 consumer CTA 之间传递 tile 级（64-256KB）中间数据。Queue 为双 buffer 设计（两个 entry），使用 sequence number 实现无锁 producer-consumer 同步。通过 CUDA API 将 queue memory pin 在 L2 cache 中，避免数据溢写到 HBM。

从kernel调度角度拆解术语：
Queue 的同步协议：

```
struct QueueEntry {
    float data[ENTRY_SIZE];    // tile数据 payload (64-256KB)
    int seq         __attribute__((aligned(128)));  // producer递增
    int consumed    __attribute__((aligned(128)));  // consumer递增
    // 全部cache-line对齐避免false sharing
};

// acquire/release API (仅CTA内threadid==0执行)

int wr_acquire(Queue* q, int tile_id):
    while true:
        seq = atomicAdd(q->seq, 0)       // 原子读取seq number
        if seq == tile_id:               // entry空闲 (可写入)
            return seq % NUM_ENTRIES     // 返回double-buffer索引
        // spin wait

void wr_release(Queue* q):
    atomicAdd(q->seq, 1)                 // 递增seq通知consumer
    __syncthreads()                      // CTA barrier: 确保所有线程完成写入

int rd_acquire(Queue* q, int tile_id):
    while true:
        seq = atomicAdd(q->seq, 0)
        if seq == tile_id + 1:           // producer已释放此entry
            return (tile_id) % NUM_ENTRIES
        // spin wait

void rd_release(Queue* q):
    atomicAdd(q->consumed, 1)            // 递增consumed释放entry
    __syncthreads()
```

Queue 性能（A100 硅片实测）：
- 无争用：100 M atomics/sec/CTA（→ 385-1541 GB/s/queue 上限）
- 54 queues（对应 108 SM）：aggregate 2 TB/s（37 GB/s/queue）@ 128-256KB payload
- 同步 overhead：12× reduction @ 1KB payload，<63% @ ≥64KB payload
- Payload > 256KB 时性能下降：queue 总大小超过 L2 capacity 溢写到 HBM（降至 1.5 TB/s）

术语一般如何实现？如何使用？
纯软件 C++ library，提供 acquire/release API。每个 CTA 仅 thread 0 执行 queue 管理操作。支持三种拓扑：1-to-1（producer-consumer pair）、1-to-many（multicast）、many-to-1（parallel reduction tree）。限制：每个 CUDA kernel 需手动改写（约 8 人时，10-40 LOC）将 global memory 读写改为 queue 读写。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---

## Implicit Fine-Grained Pipeline (ImFP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Implicit Fine-Grained Pipeline (ImFP) 是LiquidGEMM提出的W4A8 GEMM kernel pipeline执行机制。采用single-producer multiple-consumer模型：一个专门的Load WG（Warp Group）通过TMA从GMEM加载weight到SMEM并切分为fine-grained tasks，多个Compute WG竞争获取这些tasks并各自完成dequantization+MMA。关键创新：(1) 同一Compute WG负责dequantization和MMA，消除ExCP（显式粗粒度pipeline）的SMEM↔RF round-trip数据搬运；(2) dequantization和MMA的overlap通过跨Compute WG实现（WG_0做dequant时WG_1做MMA），而非同一WG内串行；(3) task scheduling由硬件管理（atomic竞争获取），无需软件barrier同步。

从kernel调度角度拆解术语：
ImFP执行流程（每thread block = 3 WGs: 1 Load + 2 Compute）：
```
// Load WG (4 warps, TMA):
for each K_tile iteration:
    TMA: GMEM → SMEM[pong]  // async weight load (Dual-MMA packed)
    cp.async.bulk.wait_group
    for each MMA fragment in tile:
        smem_task_queue.push({frag_addr, frag_meta})  // metadata only
    swap(ping, pong)

// Compute WG_0 & WG_1 (各4 warps, CUDA + Tensor Cores):
while true:
    task = smem_task_queue.try_pop()  // atomic竞争, 无barrier
    if !task: break
    LDS.128: RF = SMEM[task.addr]  // 32 UINT4, single instruction
    unpack_4bit(RF)                 // 8 elem to 2 regs
    dequant_LQQ(RF)                 // IMAD + XOR, CUDA Cores
    WGMMA(C_frag, A_frag, RF)       // INT8 MMA, Tensor Cores
// WG_0和WG_1处理不同tasks，dequant与MMA自然跨WG重叠
```

与ExCP对比：ExCP需要Load WG→Dequant WG→MMA WG三阶段，Dequant WG从SMEM读到RF dequant后写回SMEM，MMA WG再读到RF做MMA——产生SMEM↔RF round-trip × 2和barrier同步开销。

术语一般如何实现？如何使用？
基于CUTLASS/Cute warp-specialized kernel框架实现。Task queue用SMEM中的metadata数组+atomic counter。每block 1 Load WG + 2 Compute WGs（12 warps = 384 threads）。配合Dual-MMA packed layout使LDS.128充分利用带宽。消融实验：ExCP在small batch退化（round-trip+sync开销），ImFP在所有batch size持续提升。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---

## Dual-MMA Packed Layout

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-MMA Packed Layout是LiquidGEMM提出的W4A8 weight矩阵内存布局优化。核心洞察：单个WGMMA操作每个thread需要16个UINT4元素，但LDS.128指令可一次加载32个UINT4元素。将两个连续MMA操作所需元素打包存储，使每个thread用单条LDS.128加载全部32元素。与QServe的2D layout不同，Dual-MMA采用1D layout消除shared memory bank conflict，无需swizzle或复杂packing。离线变换无运行时开销。

从kernel调度角度拆解术语：
```
传统单MMA layout:
  Thread需要16 UINT4 → LDS.32加载浪费50%数据
  ldmatrix不可用 (设计为1B/element, 4-bit不兼容)

Dual-MMA Packed Layout:
  Thread需要32 UINT4 (16 MMA_0 + 16 MMA_1) → LDS.128一次加载
  元素交错排列使MMA_0/MMA_1所需数据相邻存储
  1D layout: 连续排列, 无bank conflict, 支持8路并发LDS.128
  GMEM与SMEM layout一致 (LDG.128, 离线变换)
```

术语一般如何实现？如何使用？
离线量化阶段完成layout变换写入checkpoint。CUTLASS data layout abstraction配置TMA descriptor的block shape/stride。K_tile size必须≥64（2×32）以保证双MMA打包有效。不适用：activation端（动态在线量化，无法预排列）。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---

## Asymmetric GEMM (on GPU Tensor Cores)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Asymmetric GEMM（非对称精度GEMM）指权重和激活使用不同精度的矩阵乘法，特指W4A8 GEMM（4-bit weight × 8-bit activation）。与Symmetric GEMM（W8A8, FP8）中操作数精度相同可直接输入Tensor Cores不同，Asymmetric GEMM需在main-loop中通过CUDA Cores将低精度操作数dequantize到高精度后才能执行MMA。这引入额外pipeline stage——dequantization——使用与MMA不同的硬件单元（CUDA Cores vs Tensor Cores），成为性能瓶颈或优化机会。

从kernel调度角度拆解术语：
```
Symmetric GEMM (W8A8):
  Main-loop: Load W8→SMEM → ldmatrix→RF → WGMMA(W8, A8)
  全在Tensor Core数据路径

Asymmetric GEMM (W4A8):
  Main-loop: Load W4→SMEM → LDS→RF → Dequant(W4→W8, CUDA Cores) → WGMMA(W8, A8)
  引入CUDA Core dequantization stage
  挑战: CUDA Cores 60 TFLOPS << Tensor Cores 990 TFLOPS INT8 (H100)

性能模型 (LiquidGEMM Eq.6):
  T = ⌈M/Mt⌉ · max(T_LD, α·N·K/Φ_CUDA + min(Mt,M)·2·N·K/Φ_TC)
  T_DQ = α · N · K / Φ_CUDA  ← dequant bottleneck
  消除bottleneck需: α ≤ 5.07 (memory-bound overlap) 或 α ≤ 5.05 (compute-bound)
```

术语一般如何实现？如何使用？
QServe：QoQ dequant (α≈10+) + 串行pipeline → bottleneck严重。LiquidGEMM：LiquidQuant (α≈0.875) + ImFP pipeline → dequant被有效隐藏，在所有batch size超越W8A8。计算重构Y=(WX^T)^T更好利用WGMMA m=64维度。W4A8的实用化取决于dequantization能否被pipeline有效隐藏——这是Asymmetric GEMM的核心系统挑战。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---

## Partial Fusion (Kernel Fusion Strategy for Block Low-Rank Multiplications)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Partial Fusion是针对BLR（Block Low-Rank）矩阵乘法的GPU kernel融合策略：将BLR前向中的选定相邻操作（如permutation+bmm）融合为单个Triton kernel。解决两个极端的缺陷：(1) Full fusion——将整个BLR线性层所有操作融合为单kernel，因2-D output tiling导致冗余weight加载和中间结果重计算，且1-D tiling受限于shared memory容量（仅rank≤128可行）；(2) No fusion（PyTorch baseline）——每个操作独立kernel launch，中间张量通过global memory传递→memory-bound瓶颈。

从kernel调度角度拆解术语，给出具体例子。
Monarch ② partial fusion kernel伪代码（fuse b₂↔b₁ permutation + first bmm）：

```
// 输入: X ∈ R^{n×i}分块 [n, b₁, p]; V ∈ R^{b₁×p×(r'b₂)} 已重排布
// 输出: Z' ∈ R^{n×(b₂·b₁·r')} 直接写入permuted layout

parfor b_1 in 0..b₁-1, n_tile, r_tile:
  // 计算permutation目标索引
  b_2 = (r_start : r_end) // r'                    // 确定b₂
  r'_off = (r_start : r_end) % r' + b_1 * r'       // rank偏移

  acc = zeros(t_n, t_r)
  for p_tile in 0..ceil(p/t_p)-1:
    x = X[b_1, n_s:n_e, p_s:p_e]                   // load X tile
    v = V[b_1, p_s:p_e, r_s:r_e]                   // load V tile
    acc += dot(x, v)                                // Tensor Core MMA
  
  Z'[n_s:n_e, b_2 * n * r' + r'_off] = acc         // write to permuted position
```

BLAST ④ partial fusion（消除V→S中间物化，循环b₁维度做S-weighted累加）:
```
parfor n_tile, r_tile:
  z''= zeros(b₂, t_n, t_r)
  for b_1 in 0..b₁-1:                     // 循环而非并行b₁维度
    s = S[b_1, :, r_s:r_e].view(b₂, 1, t_r)
    z' = zeros(t_n, t_r)
    for p_tile:                            // 第一个bmm (tensor core)
      x = X[b_1, n_s:n_e, p_s:p_e]; v = V[b_1, p_s:p_e, r_s:r_e]
      z' += dot(x, v)
    z'' += s * z'                          // S-weighted累加 (CUDA core)
  Z''[:, n_s:n_e, r_s:r_e] = z''
```

术语一般如何实现？如何使用？
在Triton中实现，使用autotuner选择tile sizes（t_n, t_r, t_p ∈ {32,64,128,256}）。关键约束：(1) fused kernel shared memory ≤ SM capacity；(2) 保持足够并行度（grid dim ≥ SM数量）；(3) 优先保留tensor core利用率（BLAST ⑤ > ④的原因：⑤保持全部bmm在tensor core，④将第二bmm降级到CUDA core batched outer product）。适用所有BLR压缩模型的线性层（QKVproj, Oproj, gate/upproj, downproj, c_attn, c_fc等）。不适用单token推理（n=1, memory-bound下未融合已够快）。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Permutation Fusion (in GPU Kernels for Block Low-Rank Matrices)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Permutation Fusion是将tensor维度重排（transpose/permute/reshape）与计算操作融合到单个GPU kernel中的技术。在BLR上下文中解决：Monarch和BLAST的PyTorch基线中，permutation需要独立kernel launch——创建与输入同大小的新tensor并按新顺序写入元素。当在contiguous维度排列时访问变为uncoalesced，DRAM bandwidth利用率骤降。融合的基本原理：不在global memory中物理重排数据，而在tile加载时计算permuted index并将结果直接写入目标layout。

从kernel调度角度拆解术语：
以Monarch ② fused perm+bmm为例（b₂↔b₁ + bmm with V^T = single kernel）：

```
// 需要实现: (b₁, n, r'b₂) → (n, b₂·n·r') 同时完成bmm
// 原本: 1 kernel for bmm → output [b₁, n, r'b₂]
//       + 2 kernels for permutations → 中间128MB张量
// 融合后: 1 kernel → 输出直接为 [n, b₂·n·r']

// Permutation index计算逻辑:
// 目标: 输出中位置 (n_idx, flat_rank_idx) 对应
//   b_2 = flat_rank_idx // n // r'  (or: r_range // r')
//   r'_offset = (r_range % r') + b_1 * r'
//   最终写入: Z'[n_idx, b_2 * n * r' + r'_offset]

// Triton实现: 通过program_id和range计算输出块位置
// → 计算需要加载的输入位置（inverse permutation）
// → 加载数据 → dot() → 直接写入目标位置
// 零额外kernel launch, 零额外global memory allocation
```

BLAST ⑤ transpose-based permutation消除：
```
// 转置S/U: S^T[r, b₁, b₂], U^T[r, b₂, q] (offline)
// 从左乘: [b₁, n, r] · S^T · U^T
// 每个kernel内部transpose中间输出tile
// 全部使用triton.dot()保持tensor core → 零独立permutation kernel
```

术语一般如何实现？如何使用？
Permutation fusion要求permutation是静态已知的（非data-dependent）。在Triton中通过index arithmetic实现：输出tile写入前计算permuted addresses→直接写入。对Monarch/BLAST，所有permutation都是固定维度重排，满足条件。不适用：(a) data-dependent scatter/gather；(b) 需全局reduction的操作（如softmax）。性能提升幅度取决于原permutation的数据移动量——序列越长（n越大），消除的permutation数据移动越多，speedup越显著。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Memory Layout Optimization (for Block Low-Rank GPU Kernels)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Layout Optimization是通过改变权重tensor在内存中的存储顺序（contiguous dimension ordering）来消除在线推理时permutation操作的技术。核心原理：静态权重以"对后续bmm友好的layout"存储时，原本需要独立kernel的permutation可被完全消除（数据已是目标顺序）。区别于permutation fusion（在kernel内完成重排），layout optimization将重排移到离线阶段→零运行时开销。

论文实例：(1) Monarch V重排布——V从沿b₂ first改沿r' first contiguous，消除r'↔b₂ permutation；(2) Pre-permute下游权重——Monarch输出后被静态权重消费时，pre-permute该权重rows→消除最终permutation；(3) BLAST S/U转置——离线转置S/U消除所有在线permutation。

从kernel调度角度拆解术语：
```
// Monarch ①: V重排布 (offline, one-time)
V_old: [b₁, b₂, r', p] → permute(0,2,1,3) → [b₁, r', b₂, p]
V_new: reshape(b₁, r'*b₂, p) with r' first contiguous

// 效果: 第一批bmm(X_blocks @ V_new^T) 直接产生目标layout
//       消除独立的 r'↔b₂ permutation kernel
//       消除128MB (b₁×n×r'b₂) 中间permuted tensor

// Monarch ③: Pre-permute downstream weight
// 若Y_Monarch ∈ (b₂, n, q)后接W_down @ Y
// offline: W_down rows重排为Monarch-friendly order
// → 跳过在线 (b₂,n,q)→(n,q,b₂) permutation kernel
```

术语一般如何实现？如何使用？
静态权重pre-processing：checkpoint加载后一次性转换（<1秒）。限制：(1)仅适用于静态权重，不适用动态激活；(2)改变存储格式→与下游kernel协同设计（tile尺寸需调整以保证coalesced access）；(3)layout变化后需re-tune kernel tile sizes。对于BLAST ⑤的S/U转置，关键是保持n在contiguous维度以确保后续tensor core访问效率。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Roofline Analysis (for GPU Kernel Bottleneck Diagnosis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Roofline Analysis基于roofline模型（Williams et al., 2009）的GPU kernel性能诊断方法。将kernel的arithmetic intensity（α = FLOP/Bytes）在硬件roofline上标注——由peak compute throughput（水平线）和memory bandwidth limit（斜线）组成——判断kernel是compute-bound（α ≥ α̃）还是memory-bound（α < α̃），并估算最优可达性能。论文创新应用：通过roofline分析首次系统性解释为什么BLR方法在多token推理时性能退化——尽管FLOP减半，额外中间数据移动(b×n×r)使α下降10-15×，将compute-bound的dense层推入memory-bound区域。

从kernel调度角度拆解术语：
A40 BF16 roofline参数和BLR分析：

```
// A40 BF16
Peak Compute: ≈ 150 TFLOPS; Peak Bandwidth: ≈ 696 GB/s
Breakpoint: α̃ ≈ 215 FLOP/byte

// Llama-7B Qproj (i=o=4096, r=1024, b=16)
// 单token (n=1): 所有方法α > 215 → compute-bound
//   压缩权重直接加速 (memory traffic受weight读取主导)
//
// 多token (n=1024):
//   Dense:     α ≈ 34G/34MB ≈ 994 → compute-bound ✓
//   Low-Rank:  α ≈ 17G/17MB ≈ 978 → compute-bound ✓
//   Monarch:   α ≈ 17G/138MB ≈ 123 → memory-bound ✗ (α < 215!)
//   BLAST:     α ≈ 17G/266MB ≈ 64  → strongly memory-bound ✗✗
//
// 关键: BLR的b×n×r中间张量是memory bottleneck根源
//   Monarch: +4bnr bytes, BLAST: +8bnr bytes
//   这些中间量dense baseline零开销
```

术语一般如何实现？如何使用？
Roofline分析用于三阶段：(1) profiling——通过NCU/DCGM测量实际FLOP/s和memory bandwidth→计算实测α；(2) diagnosis——对比实测α与α̃判断瓶颈类型→指导优化方向（memory-bound→减少数据移动; compute-bound→优化计算效率）；(3) 验证——优化后重测α确认是否重回compute-bound。论文用此方法不仅诊断BLR退化根因，还验证Triton kernel优化效果。限制：(a)假设零延迟和完美overlap→实际性能低于roofline预测；(b)不考虑cache→实际effective memory traffic可能因L1/L2 hit低于模型假设。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---

## Group GEMM for MoE Inference

术语是什么？
Group GEMM（Grouped General Matrix Multiplication）是MoE推理中将多个expert的矩阵乘法合并到单个统一kernel launch中并发执行的技术。每个expert的输入token batch被分配为group内的一个独立子任务（sub-task），所有子任务在同一kernel grid内并行执行。核心优势：(1) 减少kernel launch次数——所有active experts的FFN计算在单次kernel launch中完成；(2) 自然适应动态expert activation pattern——不同token激活不同expert集合，group内子任务数动态变化；(3) 通过offline profiling为不同activation pattern选择最优kernel tile sizes。

从kernel调度角度拆解术语：
以MoDES在Qwen3-VL-MoE-30B-A3B-Instruct上执行MoE FFN为例（128 experts/layer, k=8，跳过88%后平均约1 active expert/token）：

```
// Router Kernel (fused with thresholding, single launch):
r = Router(x)                              // [128] logits
π = softmax(r)
topk = topk_indices(π, k=8)                // 8 candidate expert IDs
// Branch-free thresholding:
for i in topk:
    s_i = α̃^{(l)} · π[i]
    topk[i] = (s_i < τ) ? M+1(sentinel) : topk[i]

// MoE Dispatch (filter sentinel entries):
for each expert_id in topk:
    if expert_id != M+1:
        dispatch token → expert_id's input buffer

// Group GEMM (single kernel launch):
GroupedMatMul(
    inputs: [X_e1, X_e2, ...],             // per-expert token batches
    weights: [W_gate_e1, W_up_e1, ...],    // expert FFN weights
    outputs: [gate_e1, up_e1, ...]
)
// Each expert's GEMM as independent sub-task
// Tile sizes: offline profiled for current activation pattern

// SiLU + Down Projection (also Group GEMM):
GroupedMatMul(
    inputs: [SiLU(gate_i) ⊙ up_i, ...],
    weights: [W_down_e1, W_down_e2, ...]
)

// Gather outputs, weighted sum:
for each token:
    y = Σ π_i · ExpertOutput_i
```

术语一般如何实现？如何使用？
MoDES使用custom CUDA kernel实现Group GEMM。配合offline profiling——对different representative activation patterns进行grid search确定最优tile sizes（BLOCK_M/N/K, num_warps, etc.）。Sentinel expert filtering在dispatch阶段即完成——跳过expert不分配输入buffer，不进入Group GEMM。Group GEMM性能高度依赖workload distribution的规律性——expert skipping导致的不规则子任务大小可能降低GPU利用率，但offline profiling和tile size tuning可最大化throughput。Kernel内仅需少量warp-level元素操作（masked comparison + sentinel filtering），overhead <1% of total compute time。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---

## Flash Attention (Flash-Attention 2)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flash Attention 是 Dao et al. (2022) 提出的 IO-aware 注意力算法，通过 kernel fusion 和 tiling 策略避免将完整的 N×N attention matrix 写入 GPU HBM。核心机制：(1) Tiling——将 Q/K/V 分块，逐块计算 softmax 并增量更新（online softmax），保持 SRAM 内所有计算；(2) Recomputation——反向传播时从 SRAM 中的 Q/K/V 重计算 attention matrix 而非从 HBM 读取（avoiding O(N²) HBM access）；(3) Flash Attention 2 (Dao 2023) 进一步优化了 parallelism strategy（在 sequence length 而非 batch/head 维度做并行）和 warp 级别 work partition。

从kernel调度角度拆解术语：
```
// Flash Attention 2 forward pass (simplified per attention head):
// Q,K,V ∈ R^{N×d}, block sizes B_r (rows), B_c (columns)

parfor i from 0 to ceil(N/B_r)-1:              // outer loop over Q blocks
    Q_i = Q[i*B_r : (i+1)*B_r, :]              // [B_r, d] in SRAM
    O_i = zeros(B_r, d); l_i = -inf; m_i = zeros(B_r)

    for j from 0 to ceil(N/B_c)-1:              // inner loop over K,V blocks
        K_j = K[j*B_c : (j+1)*B_c, :]; V_j = V[j*B_c : (j+1)*B_c, :]
        S_ij = Q_i @ K_j.T                      // [B_r, B_c]

        // Online softmax rescaling (update running statistics):
        m_new = max(m_i, row_max(S_ij))
        l_new = exp(m_i - m_new) * l_i + row_sum(exp(S_ij - m_new))
        O_i = diag(exp(m_i - m_new)) @ O_i + exp(S_ij - m_new) @ V_j
        m_i = m_new; l_i = l_new

    O_i = diag(1/l_i) @ O_i                     // final normalization
    store O_i to HBM
```
Flash Attention 2 改进：
- 减少非 matmul FLOPs（从 inner loop 消除 rescaling 中的 division）
- Sequence length 维度并行化（而非 batch/head），使 block 间更独立
- Warp 调度优化减少 shared memory bank conflict

术语一般如何实现？如何使用？
CUDA 实现（https://github.com/Dao-AILab/flash-attention），通过 `flash_attn_func(q, k, v, causal=True)` 调用。支持 BF16/FP16。Mordal 在 VLM alignment 训练中使用 Flash Attention-2 加速 attention computation（`vlm_kwargs` 中通过底层框架配置）。训练时 memory 节省 O(N²)→O(N)，使更大 batch/sentence length 训练可行。H100 上支持 FP8 版本（Flash Attention-3, 2024）。PyTorch 2.0+ 通过 `torch.nn.functional.scaled_dot_product_attention` 集成（自动 dispatch 到 Flash Attention backend）。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

---

## CUDA Graph

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA Graph 是 NVIDIA CUDA 10 引入的机制，允许将一系列 GPU kernel launch、memory copy 和 memory allocation 操作预录制为一个有向无环图（DAG），后续通过单次 `cudaGraphLaunch` API 调用回放整个图，消除逐个 kernel launch 时产生的 CPU-GPU 同步开销和 CUDA driver 调度开销。

CUDA Graph 的生命周期分为三个阶段：
1. **Graph Construction（图构建）**：可以使用两种方式构建图。Stream Capture 方式是通过 `cudaStreamBeginCapture`/`cudaStreamEndCapture` 包裹目标 stream 上的一组 GPU operations，CUDA runtime 自动记录这些操作及其依赖关系为图节点。Explicit API 方式是手动调用 `cudaGraphAddKernelNode`/`cudaGraphAddMemcpyNode` 等函数显式添加节点和依赖边。
2. **Graph Instantiation（图实例化）**：通过 `cudaGraphInstantiate(exec, graph)` 将图编译为可执行对象（`cudaGraphExec_t`）。实例化过程会进行静态验证、内存预分配和优化，为后续快速回放做准备。实例化是一次性开销。
3. **Graph Launch（图启动）**：通过 `cudaGraphLaunch(exec, stream)` 将整个图提交到指定 CUDA stream。与逐个 kernel launch 不同，单次 graph launch 仅产生一次 CPU-GPU 同步，图中所有 kernel 由 GPU 自主调度执行。

此外 CUDA Graph 支持更新（Update）：`cudaGraphExecKernelNodeSetParams` 允许修改已实例化图中 kernel 节点的参数（如 tensor pointers），无需重新实例化。这在 serving 场景中特别有用，因为每轮 iteration 的 KV cache 地址变化，但 kernel shapes 不变。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Nimble 中 CUDA Graph 的 AoT (Ahead-of-Time) scheduling 流程：

```
// ===== Phase 1: AoT Preparation (一次) =====
// Step 1: TorchScript tracing
model_ts = torch.jit.trace(model, dummy_input)
// → 生成包含所有算子的 static computation graph

// Step 2: Stream Assignment (Graph Rewriter)
dag = build_dag(model_ts)                   // 从 TorchScript graph 构建 DAG
meg = compute_minimum_equivalent_graph(dag)  // 计算最小等价图
bipartite = construct_bipartite(meg)         // 构建二分图
matching = ford_fulkerson_max_matching(bipartite)  // 寻找最大匹配
stream_assignment = assign_from_matching(matching) // 算子 → stream 分配
// 输出: 每个算子被分配到特定 CUDA stream (stream_0, stream_1, ...)

// Step 3: CUDA Graph Capture
for each stream in streams:
    cudaStreamBeginCapture(stream)           // 开始捕获该 stream
    // 执行分配到此 stream 的所有算子（使用 dummy input）
    for op in stream_assignment[stream]:
        op.forward(op_input)                 // PyTorch operator → CUDA kernel launch
    cudaStreamEndCapture(stream, &graph)     // 结束捕获，生成 CUDA GraphNode
// 同时记录跨 stream 的 CUDA event 同步点

// Step 4: Graph Instantiation
cudaGraphInstantiate(&exec, graph)           // 编译 CUDA Graph → 可执行对象
// 内存预分配: 所有中间 tensor 的 GPU memory 在此阶段分配
// AoT preparation 耗时: mean 0.35s, max 1.07s (NASNet-A large)

// ===== Phase 2: Runtime Inference (每次新输入) =====
// 与 PyTorch baseline 不同，完全绕过 Python/C++ framework runtime:
// ✗ 无 operator dispatch (Python/C++ autograd 查找)
// ✗ 无 output shape inference (meta-data computation on CPU)
// ✗ 无 GPU kernel selection (cuDNN implementation choosing)
// ✗ 无 kernel argument preparation (grid/block dims, strides)
// ✗ 无 per-operator memory allocation (cudaMalloc/cudaFree)

// 仅更新 input/output buffer pointers 后单次 launch:
cudaMemcpy(graph_input_buffer, new_input_data, ...)
cudaGraphLaunch(exec, stream_main)           // 单次 launch + 单次 CPU-GPU sync
// GPU 自主执行所有预录制 kernel，多个 stream 并行
```

Nimble 的关键设计决策：整个模型在一次 CUDA Graph 中录制完成（非 per-layer 或 per-iteration 录制），因为 static DL model 的 DAG 形状在输入变化时不改变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA Graph 通过 CUDA Runtime API 实现，核心函数为：
- `cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode)` — 开始捕获
- `cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph)` — 结束捕获并生成图
- `cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, ...)` — 实例化
- `cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream)` — 启动回放
- `cudaGraphExecKernelNodeSetParams(...)` — 更新 kernel 节点参数

在 LLM serving 场景中广泛应用（vLLM, SGLang, TensorRT 等）：
- Decode phase：固定 batch size + 固定 sequence length 的 decode step → CUDA Graph 预录制 → 消除数百次小 kernel 的 launch overhead
- 限制：(a) 图内所有 kernel 的 grid/block dims、shared memory 大小必须固定（静态 shapes）；(b) 不支持 dynamic control flow（conditional kernels）；(c) 内存地址变化时需通过 Update API 更新节点参数；(d) 多 shape 场景需预录制多个 graph instances，增加 GPU memory 开销
- Nimble 的特殊用法：AoT scheduling 中使用 CUDA Graph 不仅消除 launch overhead，更关键的是**完全绕过 PyTorch framework runtime**——因为 CUDA Graph 已包含所有 kernel 的完整执行拓扑，GPU 可脱离 CPU framework 自主执行

CUDA Graph 的限制（Nimble 论文指出）：不支持 dynamic neural network models（有 data-dependent control flow）；每个图 instance 需额外 GPU memory 存储 meta-data

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---

## Ahead-of-Time (AoT) GPU Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ahead-of-Time (AoT) GPU Scheduling（提前GPU调度）是 Nimble 提出的核心技术，指在模型实际推理执行之前完成全部 GPU task scheduling 流程——包括 operator dispatch、kernel selection、output shape inference、memory allocation 和 kernel argument preparation——将整个模型的 GPU 计算录制为可直接回放的执行图。运行时仅需将新输入数据拷贝到 GPU、更新 buffer pointers，然后通过单次 CUDA Graph launch 回放预录制的执行图。

与 JIT (Just-in-Time) compilation 对比：JIT 在运行时编译和优化，AoT 在首次执行前完成全部 scheduling。与 PyTorch eager mode 对比：eager mode 每轮 iteration 都重复完整的 scheduling pipeline（dispatch → shape inference → kernel selection → argument preparation → launch），AoT 将其全部前端化并一次性完成。

核心理念来源于 CUDA Stream Capture API 和 CUDA Graph：利用 "record-then-replay" 能力，将 GPU 执行与 CPU framework runtime 完全解耦。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// ================ AoT Scheduling 流程 ================

// Input: PyTorch model (e.g., NASNet-A mobile)
// Output: Executable CUDA Graph (含多 stream 分配)

// Step 1: DAG Extraction
dummy_input = torch.randn(batch_size, channels, height, width)
traced_graph = torch.jit.trace(model, dummy_input)
// traced_graph 包含:
//   - 所有算子及其类型 (Conv2d, BatchNorm, ReLU, MaxPool, Concat, ...)
//   - 算子间的数据依赖 (tensor producer → consumer)
//   - 每个算子的 input/output tensor shapes

// Step 2: Stream Assignment (详见 Stream Assignment Algorithm)
dag_nodes = extract_nodes(traced_graph)    // ~700 nodes for NASNet-A
dag_edges = extract_edges(traced_graph)     // data dependency edges
stream_plan = assign_streams(dag_nodes, dag_edges)
// Output: map<node_id, stream_id>, e.g.:
//   sep_conv_1 → stream_0, sep_conv_3 → stream_1 (并行分支)

// Step 3: Memory Plan
// 分析所有 intermediate tensor 的生命周期
// 实现 memory pre-allocation: 一个 tensor 的 memory 释放后可被后续 tensor 复用
liveness = analyze_tensor_liveness(traced_graph)
memory_plan = allocate_memory_pool(liveness)
// 与传统 PyTorch 的 per-operator malloc/free 不同:
// AoT 阶段一次分配所有需要的 GPU memory，无运行时 alloc/dealloc

// Step 4: CUDA Graph Capture + Instantiation
for each stream in streams:
    cudaStreamBeginCapture(stream)
    for op in get_ops_for_stream(stream):
        // 执行 operator → CUDA kernel launch 被记录
        op_impl = select_best_kernel(op)  // cuDNN vs PyTorch native
        op_impl(op_input, op_output)
    cudaStreamEndCapture(stream, &graph)
// 跨 stream 同步在 concat/add 等合并操作前插入 CUDA event
cudaGraphInstantiate(&exec_graph, graph)

// Step 5: Runtime Execution
// 对每个新输入:
cudaMemcpyAsync(nimble_input_buffer, new_input, ..., copy_stream)
cudaGraphLaunch(exec_graph, main_stream)
// No PyTorch framework participation — GPU 自主完成所有计算
```

AoT Scheduling 的本质转换：
```
Baseline (PyTorch eager, per-operator):
  for each operator op in model:
      CPU: dispatch(op) → infer_shape(op) → select_kernel(op) → 
           prepare_args(op) → cudaLaunchKernel(op)  ← 每 operator ~100μs CPU overhead
      GPU: execute kernel (~10μs for small kernels)
      → GPU idle while CPU schedules next op

Nimble (AoT Scheduling):
  AoT Phase (once): trace → assign_streams → capture CUDA Graph → instantiate
  Runtime (per input): cudaMemcpy + cudaGraphLaunch (single call)
      GPU: execute ALL kernels autonomously ← zero CPU overhead per operator
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AoT Scheduling 在 Nimble 中的实现基于 PyTorch TorchScript + CUDA Graph Capture API。用户接口极简（两行代码）：
```python
import nimble
nimble_model = nimble.Nimble(original_pytorch_model)  # 包装
output = nimble_model(input_tensor)                     # AoT preparation + inference
```
首次调用 forward 时触发 AoT preparation（trace + stream assignment + graph capture + instantiation），后续调用直接 replay CUDA Graph。

AoT Scheduling 的适用条件和限制：
- 适用：静态 DL 模型（无 data-dependent control flow，shapes 固定）。覆盖大多数 CNN、Transformer（inference）、ResNet、NAS 网络等 → "covers a wide range of models with practical, real-world impacts"（类比 TensorRT 的适用范围）
- 不适用：动态模型（dynamic control flow, variable-length sequences 且无法 padding）、训练中 batch size 变化循环
- 开销：AoT preparation 一次性平均 0.35s，后续无限次 amortize；额外 GPU memory 用于 CUDA Graph metadata 和 pre-allocated buffers
- 与 TensorRT/TVM 的关系：正交优化。TensorRT/TVM 做 graph optimization（operator fusion）+ kernel autotuning，Nimble 做 runtime scheduling overhead 消除。Nimble 叠加部分 fusion 后超越 TensorRT

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---

## Multi-Stream GPU Execution

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Stream GPU Execution（多流GPU执行）是将 GPU 计算任务（kernel launches）分配到多个 CUDA stream 上并行执行的技术。每个 CUDA stream 是一个 FIFO 命令队列，同一 stream 内的操作按序执行，不同 stream 间的操作可以在 GPU 上真正并发执行——前提是硬件资源（SM 数量、shared memory、register file）允许多个 kernel 同时驻留。

与 CPU 多线程的类比：一个 CUDA stream 类似一个线程——同一线程内的指令串行，多线程可并发。但 GPU 的并发约束更严格：kernel 的 thread blocks 必须竞争 SM 资源，只有资源足够时不同 kernel 的 blocks 才能同时执行。

在深度学习场景中，multi-stream 的价值在于：DL 模型的 DAG 中通常存在多条独立分支（如 NASNet cell、Inception module、multi-head attention），这些分支的算子间无数据依赖，可以并行执行。但默认的 PyTorch eager mode 将所有 kernel 提交到单一的 default stream，导致这些分支被串行化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Nimble 的 multi-stream 执行流程（以 NASNet-A Normal Cell 为例）：

```
// NASNet-A Normal Cell 的 DAG 结构:
// input → 
//   branch_0: sep_conv_5x5_1 → sep_conv_5x5_2 ─┐
//   branch_1: sep_conv_3x3_1 → sep_conv_3x3_2 ─┤
//   branch_2: sep_conv_5x5_3 → sep_conv_5x5_4 ─┤
//   branch_3: sep_conv_3x3_3 → sep_conv_3x3_4 ─┤
//   branch_4: sep_conv_5x5_5 → sep_conv_5x5_6 ─┤
//   branch_5: avg_pool ─────────────────────────┤
//   → concat (所有 branch 汇聚)

// Stream Assignment (由 Nimble Graph Rewriter 自动完成):
// stream_0: sep_conv_5x5_1 → sep_conv_5x5_2 → concat_input_0
// stream_1: sep_conv_3x3_1 → sep_conv_3x3_2 → concat_input_1
// stream_2: sep_conv_5x5_3 → sep_conv_5x5_4 → concat_input_2
// stream_3: sep_conv_3x3_3 → sep_conv_3x3_4 → concat_input_3
// stream_4: sep_conv_5x5_5 → sep_conv_5x5_6 → concat_input_4
// stream_5: avg_pool                      → concat_input_5
// ──── CUDA event sync barrier ────
// stream_0: concat → batch_norm → relu → ...

// GPU Timeline (简化):
// Time ─────────────────────────────────────────────────────────→
// SM0: |sep5_1|sep5_2|       |concat|bn|relu|...
// SM1: |sep3_1|sep3_2|       |               ...
// SM2: |sep5_3|sep5_4|       |               ...
// SM3: |sep3_3|sep3_4|       |               ...
// SM4: |sep5_5|sep5_6|       |               ...
// SM5: |avg_pool |           |               ...
//                             ↑ event sync (所有分支完成)
```

Multi-stream 在 serving/deployment 场景的实现模式：
```
// 常见模式 1: Compute-Memcpy Overlap
stream_compute: kernel_A → kernel_B → kernel_C
stream_copy:    cudaMemcpyAsync(H→D) → cudaMemcpyAsync(D→H)
// kernel_B 执行时，stream_copy 可同时进行数据传输

// 常见模式 2: Prefill-Decode Concurrency (LLM serving)
stream_prefill: prefill_layer_0 → prefill_layer_1 → ... → prefill_layer_N
stream_decode:  decode_iter_0 → decode_iter_1 → ... → decode_iter_M
// 两个 stream 的 kernel 共享 SM 资源（spatial multiplexing）

// 常见模式 3: 分支并行 (Nimble 的核心用例)
// 多个独立计算分支分布到多 stream，汇聚点 sync
stream_0: branch_A_subgraph_0 → branch_A_subgraph_1 → sync_point
stream_1: branch_B_subgraph_0 → branch_B_subgraph_1 → sync_point
// → sync (cudaEventSynchronize/cudaStreamWaitEvent)
//   → merged_subgraph
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
1. **手动编程**：`cudaStreamCreate(&s)` → `kernel<<<grid, block, shmem, s>>>()` → `cudaStreamSynchronize(s)` → `cudaStreamDestroy(s)`。需要开发者手动分析数据依赖并插入 CUDA event 同步。
2. **框架自动并行**：Nimble 通过 Graph Rewriter 自动分析 DAG 并进行 stream assignment，用户无需编写 CUDA stream/event 代码。
3. **编译器自动生成**：如 HuntKTm 通过 LLVM pass 自动发现 kernel 间数据依赖并生成多 stream 代码。

关键同步原语：
- `cudaEvent_t`：跨 stream 同步点。Producer stream 在完成计算后 `cudaEventRecord(event, stream)`，Consumer stream 在消费数据前 `cudaStreamWaitEvent(stream, event)`。
- Nimble 的最小同步原则：仅在有数据依赖的跨 stream 算子间插入 event，利用 MEG 消除冗余传递依赖。

限制和注意事项：
- NULL stream (default stream) 具有隐式全局同步语义——使用 NULL stream 会串行化所有其他 stream → 必须使用 per-thread default stream 或显式创建 non-blocking streams
- 多 stream 的加速比受限于 DAG 的逻辑并发度（logical concurrency）——如果模型大部分算子串行（如 VGG 式的线性 chain），多 stream 无加速效果
- Stream 数量过多可能导致 context switching/scheduling overhead → Nimble 通过 stream assignment algorithm 确定最优 stream 数量

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---

## Stream Assignment Algorithm (Minimum Equivalent Graph + Bipartite Matching)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Stream Assignment Algorithm 是 Nimble 自动将 DL 模型 DAG 中的算子分配到多个 CUDA stream 的算法，目标是实现最大逻辑并发度（maximum logical concurrency）且最小化跨 stream 同步次数（minimum number of synchronizations）。

算法基于图论，分四步：
1. **构建 Minimum Equivalent Graph (MEG)**：将原始 DAG 简化为最小等价图——去除冗余传递边（transitive edges）。如果 A→B→C 且 A→C 存在，则 A→C 是冗余边，去除后信息量不变但图结构简化。MEG 暴露了真正的直接依赖关系。
2. **构建 Bipartite Graph**：从 MEG 出发构建二分图。MEG 中的每条有向边 (u, v) 成为二分图中的节点；如果两条边 (u₁, v₁) 和 (u₂, v₂) 在 MEG 中不相交（无共同节点），则它们可以分配给不同 stream，在二分图中连接为一条边。
3. **Maximum Matching (Ford-Fulkerson)**：在二分图上运行最大匹配算法。每个 matching 代表一组可并行执行（无共同节点 → 无依赖冲突）的边。最大匹配的数量决定了最优 stream 数量——每个 matching 分配给一个 stream。
4. **Stream Assignment**：基于 maximum matching 结果，将 MEG 中的节点（算子）分配到对应 stream。同一条 stream 中算子按拓扑序排列，跨 stream 在 matching 边界插入 CUDA event 同步。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Algorithm: Stream Assignment for GPU Multi-Stream Execution
// Input: DAG G = (V, E) where V = operators, E = data dependencies
// Output: stream_assignment: map<v ∈ V, stream_id>

function assign_streams(G):
    // Step 1: Minimum Equivalent Graph (MEG)
    // Remove redundant transitive edges
    MEG = compute_closure(G)     // 计算传递闭包
    for each edge e = (u, v) in G:
        if exists path u → ... → v with length > 1 (excluding e):
            remove e from G       // e is a redundant transitive edge
    // G is now the MEG

    // Step 2: Construct Bipartite Graph
    // Bipartite nodes: each edge in MEG → 2 copies (left and right)
    left_nodes = {}; right_nodes = {}
    for each edge e = (u, v) in MEG:
        left_nodes.insert(L_e); right_nodes.insert(R_e)
    
    bipartite_edges = {}
    for each pair of edges e1 = (u1, v1), e2 = (u2, v2) in MEG:
        if e1 and e2 share no vertices:  // u1≠u2≠v1≠v2
            if e1 precedes e2 (no path from v2 to u1):
                bipartite_edges.insert(L_e1 → R_e2)

    // Step 3: Maximum Matching (Ford-Fulkerson)
    matching = max_matching(bipartite_graph)
    // matching = set of disjoint edges in bipartite graph
    // |matching| = number of parallelizable edge groups

    // Step 4: Assign streams
    num_streams = |matching|
    group edges by matching chain:
        // 每条 matching chain 中的 MEG edges → 同一 stream
        // 同一 matching 中的边共享无冲突 → 可放入同一 stream
    
    for each matching chain:
        stream = new CUDA stream
        for each edge e in chain (topological order):
            assign(e.source, stream)
            assign(e.target, stream)
    
    // Insert CUDA events at cross-stream dependency points
    for each e = (u, v) where stream(u) ≠ stream(v):
        insert_cuda_event(e, stream(u), stream(v))

    return stream_assignment
```

具体例子（以图 3 的简化 DAG 为例）：
```
// DAG: A → B → C → D → E
//       A ──────────→ E  (transitive edge)

// 为什么需要 MEG 而非仅用 max-flow:
// max_flow(A→E, DAG) = 1 (每条 A→E 路径都包含 edge (A,B))
// → 暗示"最大并行度 = 1" — 误导性结论!
// MEG 去除 A→E (冗余传递边) 后:
// MEG: A → B → C → D → E
// → 链路中 B 在 A 完成后才能开始，C 在 B 完成后才能开始
// → 正确反映了真正的并行度限制
//
// 论文强调: "the maximum flow of graph is trivially 1, 
// and does not give useful information for the stream assignment"
// MEG + bipartite matching 提供更精确的并行度分析
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Nimble 在 Graph Rewriter 组件中实现了该算法。实现基于 PyTorch TorchScript graph 的 IR 分析：首先从 `torch.jit.trace` 的 traced graph 提取 operator-level DAG，然后在 CPU 端执行 stream assignment（AoT preparation 阶段），最后在 CUDA Graph capture 时按分配在对应 stream 上执行各算子。

理论保证：论文证明该算法实现了 maximum logical concurrency（DAG 中可并行执行的最大算子数）且 minimum number of synchronizations（跨 stream 的 CUDA event 同步点最少）。

实际效果：NASNet-A mobile model 的 maximum logical concurrency 达到 15（最多 15 个可并行算子），multi-stream 自身贡献 up to 1.88× speedup（在 AoT scheduling 之上的额外加速）。

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---

## Framework Scheduling Overhead

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Framework Scheduling Overhead（框架调度开销）是深度学习框架（PyTorch、TensorFlow 等）在每个 DL 算子的执行过程中，CPU 端为准备 GPU kernel launch 而产生的非计算开销。Nimble 论文指出，framework overhead 并非单一来源，而是由多个串行执行的 CPU-side 步骤组成：

1. **Operator Dispatch**：Python/C++ autograd engine 根据 tensor types/shapes/devices 查找对应的 Function 对象和 kernel 实现。涉及虚函数调用、type dispatch、autograd graph 构建等多个步骤。
2. **Output Shape Inference**：在 kernel launch 之前，CPU 计算输出 tensor 的 shape（meta-data computation）。对于每个 operator，需要根据 input shapes 和 operator-specific rules 推断 output shape，用于后续 operator 的内存分配和 shape 检查。
3. **GPU Kernel Selection**：从多个 candidate kernel implementations 中选择最优的。例如 Conv 算子可能有多达 20+ 种 cuDNN algorithm candidates（implicit gemm, winograd, fft 等），auto-tuner 需要根据 shape 和 hardware 选择最优的。
4. **Kernel Argument Preparation**：准备 CUDA kernel launch 参数——grid/block dimensions、shared memory size（dynamic）、tensor strides 和 pointers。每个 kernel 的这些参数都必须从 tensor metadata 转换而来。
5. **GPU Kernel Launch**：通过 CUDA driver API (`cuLaunchKernel`) 提交 kernel 到 GPU work queue。涉及 CUDA driver 的内部数据结构和 CPU→GPU command 传输（通过 MMIO 或 PCIe）。
6. **Memory Allocation**：`cudaMalloc`/`cudaFree` 在每次 operator 执行时可能发生，尤其是中间 tensor 的生命周期管理（PyTorch 的 caching allocator 部分减轻了此开销）。

这些 overhead 串行累积：例如一个 GPU execution time 仅 10μs 的小 separable conv，其 CPU scheduling overhead 可能高达 100μs，导致 GPU 在 90%+ 的时间 idle。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PyTorch eager mode 下每个 operator 的完整 overhead 流程：

```
// 用户代码: output = model(input)
// 模型 forward pass: for each operator op in [conv1, bn1, relu1, conv2, ...]:

// ========== Operator Execution (per-operator overhead) ==========

// Step 1: Operator Dispatch (~20-30μs)
op_fn = torch._C._get_operation(op_name)       // Python→C++ dispatch
autograd_ctx = AutogradContext()                // 构建 autograd 计算图
saved_tensors = pack_for_backward(inputs)       // 保存 backward 所需中间值

// Step 2: Output Shape Inference (~10-20μs)
output_dims = op_fn.infer_output_shape(input.shape)
output_dtype = input.dtype  // 通常与 input 相同
output_layout = infer_memory_layout(op, input)

// Step 3: Kernel Selection (~5-50μs, depends on operator type)
// 例如 Conv2d: cuDNN 提供多种 algorithm
algorithms = cudnnFindConvolutionForwardAlgorithm(
    input_desc, weight_desc, conv_desc, output_desc, 
    max_algorithms=20
)
best_algo = min(algorithms, key=lambda a: a.time)  // auto-tune
kernel_fn = algorithms[best_algo].kernel

// Step 4: Argument Preparation (~5-10μs)
grid_dim = compute_grid(output_dims, block_dim)
block_dim = min(block_dim, MAX_THREADS_PER_BLOCK)
shared_mem = estimate_shared_memory(op, block_dim)
kernel_args = pack_kernel_launch_params(grid_dim, block_dim, shared_mem, 
                                         input_ptr, weight_ptr, output_ptr, 
                                         strides, padding, ...)

// Step 5: Kernel Launch (~3-5μs)
cudaLaunchKernel(kernel_fn, grid_dim, block_dim, kernel_args, shared_mem, stream)

// Step 6 (implicit): Memory Allocation (amortized by caching allocator)
// PyTorch caching allocator reduces malloc/free cost but still has pool lookup overhead
// New intermediate tensors still require allocator round-trips

// ========== GPU Execution ==========
// GPU executes kernel: ~10μs (small separable conv)
// CPU idle waiting for GPU (or processing framework overhead for next op)

// === Summary for ~700 operators (NASNet-A mobile) ===
// GPU compute time: ~700 * 10μs = 7ms
// CPU overhead: ~700 * 100μs = 70ms
// GPU idle ratio: 70ms / (7ms + 70ms) ≈ 91%
// → PyTorch baseline measured GPU idle up to 91%
```

Nimble 的 AoT scheduling 如何消除这些 overhead：

```
// AoT Scheduling: 所有 overhead 发生在 AoT preparation 阶段 (一次)
// Runtime: 每个 operator 的执行被简化为 CUDA Graph 中的节点重放

// Per-operator overhead 的消除:
// Step 1 (Operator Dispatch):     ✗ eliminated — CUDA Graph 已包含 kernel 引用
// Step 2 (Shape Inference):       ✗ eliminated — shapes pre-determined in AoT
// Step 3 (Kernel Selection):      ✗ eliminated — kernel pre-selected in AoT
// Step 4 (Argument Preparation):  ✗ eliminated — args pre-recorded in graph nodes
// Step 5 (Kernel Launch):         → single cudaGraphLaunch (not per-op)
// Step 6 (Memory Allocation):     ✗ eliminated — memory pre-allocated in AoT

// Runtime cost per inference:
// cudaGraphLaunch overhead: <100μs total (vs baseline's 70ms CPU overhead)
// GPU execution: ~7ms (same as baseline)
// → 22x speedup for NASNet-A mobile
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
除 Nimble 的 AoT scheduling 外，业界也存在其他减少 framework overhead 的方法：

1. **Memory Pre-allocation**：PyTorch caching allocator、TensorFlow memory planner 减少 runtime alloc/free，但仅解决 Step 6。
2. **CUDA Graph (per-step)**：vLLM 和 SGLang 将 decode step 录制为 CUDA Graph，消除 decode iteration 的 launch overhead。但仅适用于 shapes 固定的重复性步骤。
3. **TorchScript / torch.compile**：通过 JIT compilation 和 operator fusion 减少 total operator count（从而减少 dispatch 次数），但不能消除每个 operator 的 dispatch。
4. **Framework Redesign**：如 TensorFlow XLA 将整个计算图编译为单一可执行对象。但论文指出 "Redesigning the framework to remove all sources is very challenging"，而且仍有 JIT compilation 开销。

Nimble 的关键 insight：与其逐个修复 overhead 来源，不如**完全绕过** framework runtime——利用 CUDA Graph 的 record-then-replay 能力，将 GPU 执行与 CPU framework 解耦。

影响程度取决于模型特性：当模型由 large kernels（大矩阵乘、大 conv）组成时，framework overhead 占比小 → limited speedup（如 BERT training、ResNet-50 ImageNet training）。当模型包含大量 small kernels（mobile-optimized CNNs、NAS architectures 等）时，framework overhead 主导 → dramatic speedup (up to 22x)。

涉及论文标题：
- Nimble: Lightweight and Parallel GPU Task Scheduling for Deep Learning

---

## Inter-SM Overlapping (Multi-GPU Kernel Scheduling)

术语是什么？
Inter-SM overlapping将GPU的SM分区为两组：compute SM执行HBM→SMEM→tensor core计算流水线，communication SM独立执行inter-GPU数据传输或collective操作。与intra-SM（同一SM内分warp）不同，inter-SM让通信独立于计算数据流，使in-network reduction和remote cache-friendly批量传输成为可能。ParallelKittens通过LCSC template的num_comm_sms参数控制SM分区。

从kernel调度角度拆解术语：
GEMM+AR fused kernel的inter-SM执行：
```
Compute SMs (num_SMs - num_comm_sms):
  loader:   tma::load_async A, B tiles → SMEM (pipelined)
  consumer: warpgroup::mma_AB accumulate C
  storer:   tma::store_async to local HBM → signal barrier

Communication SMs (num_comm_sms):
  wait barrier(NUM_DEVICES)           // 所有GPU local compute完成
  __syncthreads()
  all_reduce<ADD>(G.C, coord)         // NVSwitch in-network reduction via multimem.ld_reduce
```
trade-off: inter-SM牺牲部分SM做通信（tensor core利用率降低），但：(1) 利用in-network reduction将all-reduce通信量从O(N) peer写入降为O(1) multicast读归约，GEMM+AR加速3.62x；(2) Ring Attention中通信SM批量传输KV block到local HBM复用L2 cache，避免per-block重复remote访问；(3) HBM同步延迟~832ns vs intra-SM mbarrier ~64ns。最优num_comm_sms与问题大小相关，PK运行时自动搜索。

术语一般如何实现？如何使用？
PK: lcsc::launch_kernel + num_comm_sms配置。适用：(a) 利用in-network reduction的all-reduce/collective；(b) 需要remote cache reuse的Ring Attention；(c) 通信模式与计算模式不对齐的场景。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---

## Intra-SM Overlapping (GPU Compute-Communication Fusion)

术语是什么？
Intra-SM overlapping在同一SM内并发执行compute和inter-GPU通信：利用TMA单线程异步特性，一条线程发出TMA通信指令后立即返回，其余warp同时执行tensor core WGMMA指令。所有SM的所有tensor core保持繁忙，通信在后台进行。ParallelKittens推导出BF16 GEMM+RS on H100的完全隐藏条件：K ≥ sR/(2B) ≈ 2197，实测K=4096时non-overlapped communication <1%。

从kernel调度角度拆解术语：
GEMM+RS时间线（intra-SM，单SM视角）：
```
Warp0(loader,1 thread):  | TMA load A_tile | wait | TMA load next | ...
Warp1-3(consumer):       | WGMMA C+=A×B     | WGMMA | WGMMA ...
Warp4(storer,1 thread):  | TMA store_add_async to peer | wait | ...
                           ↑ TMA不占用tensor core，单线程异步
```
优势：所有tensor core满利用率；mbarrier同步仅~64ns延迟。局限：(1) 通信跟随计算数据流，无法利用in-network reduction；(2) 对remote cache unfriendly场景（Ring Attention），重复remote访问快速饱和NVLink带宽。

术语一般如何实现？如何使用？
PK: TMA store_async + store_add_async原子加到peer PGL。适用：(a) 通信模式与计算模式对齐（GEMM+RS——每个tile写入唯一目标）；(b) 通信量可被计算覆盖的大K场景（K>2197）；(c) 不需要in-network acceleration的P2P传输。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---

## Parallel Global Layout (PGL)

术语是什么？
Parallel Global Layout (PGL)是ParallelKittens的多GPU核心数据结构，表示所有参与GPU上具有相同shape/size的HBM内存区域集合。PGL封装multicast memory（VMM创建）和IPC-exported memory的统一寻址，以tile-indexed坐标(int4: {batch, depth, row, col})访问。PGL是PK 8种原语中P2P通信（store_async, store_add_async）和in-network collective（reduce, all_reduce）的操作目标。

从kernel调度角度拆解术语：
PGL操作示例：
```
// tile坐标: int4{b, d, r, c}
store_async(dst_PGL, src_stile, coord)           // TMA异步存储到multicast memory
store_add_async(dst_PGL, src_stile, coord)       // TMA原子加(实现reduce-scatter)
reduce<ROW, COL, OP::ADD>(dst_local, d_coord, src_PGL, s_coord) // in-network reduction local
all_reduce<ROW, COL, OP::ADD>(PGL, coord)        // in-network all-reduce
```
address duality: local address（写到本GPU物理HBM）vs multicast address（写到multicast object → NVSwitch broadcast；读+multimem.ld_reduce → in-network reduction）。PGL自动处理coalesced NVLink access和tensor core-friendly swizzle。

术语一般如何实现？如何使用？
PK utility层在kernel启动前完成VMM分配+fd交换+multicast create+bind+map，kernel内仅通过coord寻址PGL。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---

## LCSC Programming Template

术语是什么？
LCSC (Load-Compute-Store-Communicate) Template是PK的统一多GPU kernel编程模板，分解kernel为4个worker：Loader (TMA HBM→SMEM)、Storer (TMA SMEM→HBM)、Consumer (tensor core compute)、Communicator (专用通信SM上的inter-GPU collective)。模板自动处理SM/warp分区、SMEM管理、barrier同步、TMA设置，用户仅需实现per-tile逻辑。每个kernel通信代码<50行。

从kernel调度角度拆解术语：
```
struct lcsc_template {
    static void loader(globals, comp_sem, comp_smem, comp_regs);
    static void storer(globals, comp_sem, comp_smem, comp_regs);      // 含signal barrier
    static void consumer(globals, comp_sem, comp_smem, comp_regs);    // warpgroup MMA
    static void communicator(globals, comm_sem, comm_smem, comm_regs); // wait+all_reduce
};
lcsc::launch_kernel<config, globals, lcsc_template>(G, stream);  // host launch
```
执行模型：compute SM运行loader+consumer+storer（producer-consumer semaphore同步），communication SM运行communicator。config编译时确定SM/thread/warpgroup配置，num_comm_sms控制inter-SM vs intra-SM切换（communicator为空即intra-SM模式）。

术语一般如何实现？如何使用？
PK所有6个kernel（AG+GEMM, GEMM+RS, GEMM+AR, Ring Attention, Ulysses, MoE dispatch+GEMM）均用LCSC模板实现。GEMM+AR的通信代码仅~10行。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

## Block Sparse Attention (GPU Block-Level)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Sparse Attention是将注意力稀疏性从element-wise提升到block level的实现策略，使稀疏注意力在现代GPU上高效执行。动机：element-wise sparse attention（在单个Q_i·K_j dot-product级别做mask）在GPU上效率低——不规则的稀疏pattern导致warp divergence、uncoalesced memory access和低tensor core利用率。Block-level sparse attention将Q、K、V、S、P、M都划分成blocks {Q_i}、{K_j}、{V_j}，每个block mask M_{ij} ∈ {0,1}^{b_q×b_{kv}}完全填0或1，跳过M_{ij}=0的整块Q_i K_j^T和P_{ij} V_j计算。

在SLA中，block sparse attention用于critical块（M_c[i,j]=1, ~5% blocks）。每对critical (Q_i, K_j, V_j)执行完整FlashAttention：S_{ij} = Q_i K_j^T/√d, OnlineSoftmax normalization, O_i^s += P_{ij} V_j。block size b_q=b_{kv}=64是在GPU效率和分类粒度间的平衡——太小的block导致mask预测开销增大，太大的block使分类粗糙。

从kernel调度角度拆解：
```
Block Sparse Attention Forward (per Q block Q_i):
  for j in 0..T_n-1:
      if M_c[i,j] == 1:   // critical block pair
          // Full FlashAttention on this block:
          S_ij = Q_i @ K_j^T / sqrt(d)    // [b_q, b_{kv}] GEMM, Tensor Cores
          // OnlineSoftmax rescaling:
          m_new = max(m_prev, rowmax(S_ij))
          P_ij = exp(S_ij - m_new)        // [b_q, b_{kv}]
          l_new = exp(m_prev - m_new)*l_prev + rowsum(P_ij)
          O_i_s = exp(m_prev - m_new)*O_i_s + P_ij @ V_j  // [b_q, d] GEMM
          m_prev = m_new; l_prev = l_new
      // else: skip entire block computation
  O_i_s = diag(1/l_prev) @ O_i_s  // final normalization
```

关键效率考量：block-level sparsity使tensor core GEMM操作均在规则的[b_q, b_{kv}]或[b_q, d] tile上执行，无warp divergence。SLA的M_c在block级别分类（分辨率为N/b_q × N/b_{kv} = 469×469 for N=30K），而非元素级别（N×N = 30K×30K），使mask预测和存储开销可忽略。

术语一般如何实现？如何使用？
Block sparse attention的GPU实现通常基于FlashAttention框架（https://github.com/Dao-AILab/flash-attention），在tiling outer loop中插入block mask检查。SLA将block sparse attention与linear attention融合在单个kernel中。Block size选择：b_q=b_{kv}=64是FlashAttention的典型block大小，平衡SRAM使用和并行度。VSA、VMoBa等方法也使用类似的block-level稀疏策略，但分类粒度（block vs element）和mask预测方法（训练式 vs 训练无关）不同。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## OnlineSoftmax (Block-Wise Incremental Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OnlineSoftmax (Milakov & Gimelshein, 2018) 是一种在矩阵分块场景下增量计算softmax的算法，是FlashAttention实现的核心技术之一。标准softmax需要三次pass over the entire row：一次求max（numerical stability）、一次求exp sum、一次求normalized values。OnlineSoftmax将各行分块，逐block增量更新running max和running sum，每块仅需一次前向pass，最后做一次归一化。核心更新公式：

$$m_{new} = \max(m_{old}, \operatorname{rowmax}(S_{block})), \quad l_{new} = e^{m_{old}-m_{new}} \cdot l_{old} + \operatorname{rowsum}(e^{S_{block}-m_{new}})$$

$$O_{new} = \operatorname{diag}(e^{m_{old}-m_{new}}) \cdot O_{old} + e^{S_{block}-m_{new}} \cdot V_{block}$$

在SLA中，OnlineSoftmax用于critical块（M_c[i,j]=1）的稀疏FlashAttention计算。由于critical块在K维度是非连续的（被marginal和negligible块间隔），OnlineSoftmax的增量更新特性允许在遍历K块时自然地跳过非critical块（无需重新归一化已完成的部分）。

从kernel调度角度拆解：
```
OnlineSoftmax in SLA critical block processing:
  m_prev = [-inf, -inf, ..., -inf]  // per-row running max, [b_q]
  l_prev = [0, 0, ..., 0]          // per-row running sum, [b_q]
  O_i_s = zeros(b_q, d)            // running weighted output

  for j where M_c[i,j] == 1:       // only critical K,V blocks
      S_ij = Q_i @ K_j^T / sqrt(d)  // [b_q, b_{kv}]
      m_curr = elementwise_max(m_prev, rowmax(S_ij))
      
      // Rescale old accumulators to new max:
      scale = exp(m_prev - m_curr)  // [b_q]
      l_curr = scale * l_prev + rowsum(exp(S_ij - m_curr))
      O_i_s = diag(scale) @ O_i_s + exp(S_ij - m_curr) @ V_j
      
      m_prev = m_curr
      l_prev = l_curr

  // After ALL critical blocks processed:
  O_i_s = diag(1/l_prev) @ O_i_s   // final normalization
```

关键特性：marginal块的线性注意力（H_i += h_j）不参与softmax归一化——marginal块用独立的线性注意力路径，critical块用OnlineSoftmax归一化，两者在最终输出时通过Proj融合。这使得critical块间的非连续遍历不影响OnlineSoftmax的正确性。

术语一般如何实现？如何使用？
OnlineSoftmax是FlashAttention (Dao et al., 2022; Dao, 2023)的标准实现组件。在CUDA kernel中以register-resident的m和l向量实现（每行一个scalar，存储在warp-level registers）。SLA复用FlashAttention的OnlineSoftmax实现用于critical块，并在同一kernel中添加线性注意力路径。FlashAttention 2 (Dao, 2023) 进一步优化了OnlineSoftmax的rescaling开销（减少非matmul FLOPs）。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## Fused Sparse-Linear Attention GPU Kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused Sparse-Linear Attention Kernel是SLA的核心实现——将稀疏FlashAttention（O(N²) per critical block）、线性注意力（O(1) per marginal block via precomputation）和negligible block skipping三种不同计算模式融合到单个CUDA kernel中，支持完整的前向和反向pass。融合的关键优势：(1) 单次kernel launch消除多次launch overhead；(2) 所有block计算共享同一Q/K/V数据加载，避免重复HBM访问；(3) 前向和反向对称设计，反向也融合sparse gradients和linear gradients。

从kernel调度角度拆解：
```
Fused SLA Forward Kernel (Algorithm 1, single CUDA launch):

// Phase 1: Precompute for linear attention (GPU parallelized over j)
parfor j in 0..T_n-1:
    K_phi_j = softmax(K_j)           // per-block activation
    h_j = matmul(K_phi_j^T, V_j)     // [d,d] — stored in HBM for all Q blocks
    z_j = rowsum(K_phi_j^T)          // [d,1]

// Phase 2: Main fused loop (GPU parallelized over i)
parfor i in 0..T_m-1:
    O_i_s, H_i, Z_i = 0, 0, 0
    m_prev, l_prev = -inf, 0
    
    for j in 0..T_n-1:
        if M_c[i,j] == 1:   // CRITICAL → Tensor Core GEMM pipeline
            S_ij = wgmma(Q_i, K_j^T) / sqrt(d)     // Tensor Cores
            m_curr, l_curr = online_softmax_update(S_ij, m_prev, l_prev)
            O_i_s = rescale_and_accumulate(O_i_s, S_ij, V_j, m_prev, m_curr)
            m_prev, l_prev = m_curr, l_curr
        elif M_c[i,j] == 0: // MARGINAL → CUDA Core addition
            H_i += h_j      // d×d matrix addition
            Z_i += z_j      // d×1 vector addition
        // else NEGLIGIBLE → no operation

    O_i_s = diag(1/l_prev) @ O_i_s       // normalize sparse output
    Q_phi_i = softmax(Q_i)
    O_i_l = (Q_phi_i @ H_i) / (Q_phi_i @ Z_i)  // linear output

// Phase 3: Fusion output
O = O_s + Proj(O_l)  // learnable projection
```

反向kernel（Algorithm 2）对称融合：sparse gradients遵循FlashAttention backward公式（dO^s → dS_{ij} → dQ_i, dK_j, dV_j），linear gradients通过chain rule（dO^l → dH_i/dZ_i → dQ_i^φ, dK_j^φ, dV_j），marginal块的梯度聚合也为矩阵加法（dH_agg += dH_i, dZ_agg += dZ_i）。

性能特征：Forward 13.7× vs FlashAttention2 (RTX 5090, Wan2.1-1.3B)；Backward 6.8× vs FlashAttention2。marginal块（~85% of blocks）仅占<0.5% full attention cost，使得critical块的Tensor Core GEMM主导执行时间，GPU利用率接近dense attention但计算量仅5%。

术语一般如何实现？如何使用？
实现为CUDA kernel（论文未开源实际kernel代码，仅开源高层接口 https://github.com/thu-ml/SLA）。使用WGMMA (warp group matrix multiply-accumulate) for Tensor Core GEMM on critical blocks, TMA for async data loading, CUDA cores for linear attention additions。额外优化（Appendix A.3）：Lookup table存储非零mask位置（sparsity>90%）、Pre-aggregation（用∑h_j - ∑_{M_c≠0}h_j替代逐个加法）、Method of Four Russians（分组预计算子集和）。块大小b_q=b_{kv}=64（已在线性注意力预计算开销和分类粒度间平衡优化）。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## Method of Four Russians (in SLA GPU Kernel Context)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Method of Four Russians (Arlazarov et al., 1970) 是布尔矩阵乘法和传递闭包计算的经典算法，核心理念是通过预计算和查表来减少在线计算量。SLA在Appendix A.3中将其适配到marginal块的线性注意力聚合场景：当M_c中标记为0（marginal）的块数量既不太小也不太大（~50%）时，将连续的g个h_j和z_j分组，预计算每组内所有2^g种可能的子集和存入查找表，前向/反向时任何子集的聚合结果通过单次查表获取，而非on-the-fly逐个求和。

从kernel调度角度拆解：
```
Standard marginal block aggregation (line 13 in Algorithm 1):
  for each Q block i:
      H_i = 0
      for j where M_c[i,j]==0:
          H_i += h_j    // one d×d addition per marginal block
  // Cost: (#marginal_blocks) × (d×d addition)

Method of Four Russians optimization:
  // Offline/precompute: group h_j into segments of g consecutive blocks
  for each segment s of g blocks:
      precompute all 2^g subset-sums of h_j in segment s
      store in lookup table LUT_s[bitmask]  // 2^g entries, each d×d

  // Online: use lookup table
  for each Q block i:
      H_i = 0
      for each segment s:
          bitmask = extract_g_bits(M_c[i, segment_range])
          if bitmask != 0:  // not all negligible/critical
              H_i += LUT_s[bitmask]    // single lookup + addition
  // Cost: (#segments) × (one lookup + one addition)
  // Theoretical reduction: 1/g
```

适用条件：marginal块比例~50%时最优。sparsity极高（>90%）时用Pre-aggregation更优（∑_all - ∑_non_marginal），sparsity极低时直接加法即可。SLA中默认使用直接加法（85% marginal块），Method of Four Russians作为备选优化。

术语一般如何实现？如何使用？
经典实现用于布尔矩阵乘法和transitive closure。在SLA的GPU kernel中：预计算的lookup table存储在GPU global memory或shared memory中（取决于g大小），g的选择平衡查找表大小（2^g × d×d，exponential in g）和计算节省（1/g reduction）。SLA论文未报告此优化的独立ablation结果，将其列在Appendix A.3作为supplementary efficiency optimization。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention

## Producer Warp Epilogue

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Producer Warp Epilogue 是 SageAttention3 中针对 FPGA 约束下的 warp-specialized kernel 提出的一种新型 warp 调度策略。传统 warp-specialized GEMM kernel（如 CUTLASS 高效 GEMM）中，consumer warp 负责 Tensor Core MatMul + accumulator store to global memory，producer warp 仅负责从 global memory 加载数据到 shared memory，consumer 之间做 ping-pong 重叠（while one consumer computes, another consumer stores）。然而 FP4 attention kernel 的寄存器压力极高（因 FP4MMA 的 register fragment layout 复杂 + online softmax 状态 + two-level quantization 中间值），consumer warp 同时承担 MatMul 和 store 会导致寄存器溢出（register spilling），严重拖慢性能。Producer Warp Epilogue 将 store 职责从 consumer 移交给 producer：使用两个 producer warp 做 ping-pong — 一个 producer 加载下一轮输入数据时，另一个 producer 存储上一轮 MatMul 输出到 global memory。Consumer warp 仅负责将 FP4MMA 结果从寄存器搬运到 shared memory（低开销，寄存器需求少）。此设计在受限寄存器条件下实现 MatMul 与 global store 的 overlap。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 传统 warp-specialized kernel (consumer-producer)
producer_warp: while has_tile:
    load Q/K/V tile → shared_memory
    signal consumer

consumer_warp_0, consumer_warp_1 (ping-pong):
    while has_tile:
        wait producer signal
        FP4MMA QK^T  // consumer does MatMul
        softmax + two-level quantize P
        FP4MMA PV     // consumer does MatMul
        store O to global_memory  // consumer does store ← register pressure!

// Producer Warp Epilogue (SageAttention3)
producer_warp_0, producer_warp_1 (ping-pong):
    while has_tile:
        // Phase A: Load next tile
        load Q/K/V tile → shared_memory  // producer_warp_0
        // Phase B: Store previous output (overlapped with consumer MatMul)
        store O_prev (in shared_memory) → global_memory  // producer_warp_1
        signal consumer

consumer_warp:
    while has_tile:
        wait producer signal
        FP4MMA QK^T           // consumer only does MatMul
        softmax + quantize P
        FP4MMA PV
        move O to shared_memory  // consumer only moves to SMEM (lightweight)
        signal producer
```
关键区别：consumer 不再直接 store 到 global memory，而是将结果放入 shared memory；producer 在加载下一批数据的同时执行 store。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在 CUDA C++ 中使用协作组（cooperative groups）或显式 warp 级别同步（`__syncwarp()`）实现。Producer warp_0 和 producer_warp_1 通过 shared memory flag 同步，consumer warp 通过 producer-consumer barrier 同步。Shared memory 用作 producer-consumer 之间的 output buffer。此优化带来的 kernel 加速约 10%（与 reuse shuffle 一起）。适用于所有寄存器压力大、传统 consumer-producer 分工无法容纳完整 pipeline stage 的 warp-specialized kernel。不适用于寄存器充裕的简单 kernel（此时传统方案更简单高效）。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

## WGMMA (Warp Group Matrix Multiply-Accumulate)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WGMMA（Warp Group Matrix Multiply-Accumulate）是NVIDIA Hopper架构的Tensor Core PTX指令集，由warpgroup（128线程）协同发起矩阵乘法累加操作。相比Ampere架构的wmma指令（由单个warp的32线程发起），WGMMA代表了Tensor Core编程的根本性变革：(1) 更大的tile尺寸——WGMMA支持如64×256×16、64×128×16等更大MMA shape（vs Ampere的16×16×16），提供更高的算术强度；(2) 操作数来源——操作数A/B可来自线程寄存器或shared memory的任意组合（寄存器-寄存器、寄存器-SMEM、SMEM-寄存器），由指令变体决定；(3) 异步执行——WGMMA是异步指令，发出后线程可继续执行其他工作（如准备下一tile的操作数），通过`wgmma.wait_group`等待完成；(4) 复杂的数据分布——输出矩阵C和操作数A/B按照硬件规定的swizzle pattern分布在128线程的寄存器和shared memory中（见Cypress论文Figure 4的64×N×8 partition pattern）。

Cypress论文的gemm_thread leaf task直接使用WGMMA：
```
CuTe_warpgroup_gemm(WGMMA_64x256x16(), C, A, B)
```
其中WGMMA_64x256x16()是64×256×16的MMA shape参数化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WGMMA在Hopper GEMM kernel中的调度伪代码（compute warpgroup部分）：

```
// Compute warpgroup: 128 threads, 4 warps
// 输入: accum[8] (per-thread accumulator registers, swizzled layout)
//       sA[T_M, T_K, PIPE] (shared memory, 3-deep pipeline)
//       sB[T_K, T_N, PIPE]

pipeline_depth = 3
for k = 0 to K/T_K - 1:
    wait(prod[k % pipeline_depth])      // 1. 等待TMA完成本iteration的数据加载
    
    warpgroup_sync()                     // 2. 128线程对齐
    
    wgmma.fence                         // 3. 确保A/B操作数就绪
    wgmma(accum,                        // 4. 异步发起Tensor Core计算
          sA[:, :, k % pipeline_depth],  //   操作数A来自SMEM
          sB[:, :, k % pipeline_depth])  //   操作数B来自SMEM
    // 线程可继续其他工作（如address计算等）
    
    warpgroup_wait()                     // 5. 等待Tensor Core完成
    // accumulator已更新：C += A_tile * B_tile
    
    arrive(cons[k % pipeline_depth])    // 6. 通知DMA warp buffer可重用

// 最终：warpgroup将accumulator从寄存器写回shared memory staging buffer
//       DMA warp通过TMA_store将结果写入global memory
```

WGMMA指令的关键参数：
- m64nNk16/m64nNk32: m=64固定（warpgroup的行粒度），N可变，k=16或32
- operand A来源：register或shared memory
- operand B来源：register或shared memory  
- 输出accumulator：始终在warpgroup各线程的寄存器中（不可直接写shared memory或global memory）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 直接编写PTX汇编（`wgmma.fence.sync.aligned`, `wgmma.mma_async.sync.aligned`等）
- CUTLASS/CuTe封装：`cute::gemm`或`cute::gemm_rs`（register-smem variant）
- ThunderKittens提供高级C++封装
- 需要与thread block配置匹配——block size必须为warpgroup size的倍数（≥128）
- WGMMA的使用通常与warp specialization协同——DMA warp管理TMA传输，compute warpgroup专注于WGMMA

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs

## Warp Specialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Warp Specialization是一种GPU编程技术，将thread block内的不同warps分配为不同的角色（如数据搬运、计算、输出写入），利用GPU warp调度器的细粒度warp切换实现不同角色之间的overlap执行。在Hopper架构上，warp specialization是实现高性能的必需手段——因为TMA和Tensor Core都是异步固定功能单元，需要不同的warp专门管理各自的操作以最大化硬件利用率。

Cypress论文详细描述了Hopper上的warp specialization模式：(1) DMA Warp——1个warp（32线程），专门执行TMA异步数据搬运（实际仅thread 0调用TMA指令），其余线程可能执行地址计算或闲置；(2) Compute Warpgroup——4个warp（128线程），专门执行WGMMA Tensor Core操作；(3) 寄存器复用——DMA warp几乎不消耗寄存器，其寄存器资源可通过硬件bank分配到compute warpgroup，允许存储更大的accumulator或更多的pipeline stages。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hopper GEMM中warp specialization的调度伪代码（完整CTA=128+32=160 threads）：

```
// ─── DMA Warp (32 threads, warp_id=4) ───
TMA_dma_warp():
    for k = 0 to K/T_K - 1:
        // Pipelining: DMA领先compute PIPE步
        if k >= pipeline_depth:
            wait(cons[k % pipeline_depth])  // 等待consumer释放buffer
        if thread_id == 128:               // 仅1个线程调用TMA
            TMA_load(
                completion_barrier = prod[k % pipeline_depth],
                src  = gA[:, :, k],       // global memory tile
                dst  = sA[:, :, k % pipeline_depth])  // shared memory
            TMA_load(
                completion_barrier = prod[k % pipeline_depth],
                src  = gB[:, :, k],
                dst  = sB[:, :, k % pipeline_depth])
    // 最终：等待consumer完成后写output
    wait(copyout)
    if thread_id == 128:
        TMA_store(sC → gC[blk_x, blk_y])

// ─── Compute Warpgroup (128 threads, warp_id=0,1,2,3) ───
TMA_compute_warpgroup():
    for k = 0 to K/T_K - 1:
        wait(prod[k % pipeline_depth])     // 等待TMA完成
        warpgroup_sync()                    // 128线程对齐
        wgmma(accum, sA[:,:,k%P], sB[:,:,k%P])  // Tensor Core
        warpgroup_wait()                    // 等待Tensor Core完成
        arrive(cons[k % pipeline_depth])   // notify DMA: buffer free
    
    // Epilogue: accumulator registers → shared memory → TMA store
    copy_reg_to_smem(accum → sC)
    syncthreads()
    arrive(copyout)
```

关键调度特征：
- DMA warp和compute warpgroup交替执行（warp scheduler自动time-multiplexing）
- 当compute wg在等Tensor Core完成（warpgroup_wait）时，warp scheduler自动切换到DMA warp
- 反之，当DMA warp在等consumer释放buffer（wait(cons)）时，compute warpgroup获得执行
- Pipelining (PIPE=3)确保TMA延迟被完全隐藏

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- CUTLASS 3.x：warp-specialized main loop模板（`sm90_mma_tma_gmma_rs_warpspecialized.hpp`）
- ThunderKittens：通过LCSF模板将warp specialization形式化为Load/Compute/Store/Finish四个角色——load worker (warp 0) 执行 TMA 异步 load，compute workers (warpgroup) 执行 tile 计算，store worker 执行 TMA store，finish 处理退出。用户只需填充每个角色的函数体，框架自动管理 barriers 和 pipeline buffer。对比 CUTLASS 中手动管理 ping-pong scheduler 的 warp specialization，TK 的 LCSF 将 attention 实现从 2325 行减至 217 行，同时消除 FA3 的 9.6-way bank conflict。
- CUDA C++手动实现：使用`cooperative_groups::tiled_partition`和显式barrier管理
- 关键难点：(1) 正确管理producer-consumer barriers（避免deadlock和data race）；(2) 寄存器分配——误用导致spilling严重；(3) shared memory banking——pipeline buffers需正确对齐

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
- ThunderKittens: Simple, Fast, and Adorable Kernels

## Producer-Consumer Pipeline (Async GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Producer-Consumer Pipeline（生产者-消费者流水线）是GPU上利用异步固定功能单元实现计算与数据搬运重叠的软件pipeline技术。在Hopper GPU上，producer（DMA warp通过TMA生产数据到shared memory）和consumer（compute warpgroup通过WGMMA消费shared memory中的数据并产出accumulator）通过named barriers连接，形成深度为PIPE的流水线——producer领先consumer PIPE步预取数据，使TMA的global memory访问延迟被完全隐藏在consumer的计算时间中。

Cypress论文展示的pipeline结构具有三个关键同步原语：
1. prod barriers：TMA完成数据加载后自动arrive，通知consumer数据就绪
2. cons barriers：consumer完成计算后arrive，通知producer该buffer可安全被新数据覆盖
3. copyout barrier：最终output staging完成，通知DMA warp可TMA_store到global memory

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
3-deep Producer-Consumer Pipeline在Hopper GEMM中的执行时间线：

```
PIPE=3, 每个SM执行5个K-reduction迭代

Timeline (时间从左到右):
─────────────────────────────────────────────────────────────────
Iter: k=0       k=1       k=2       k=3       k=4      Epilogue
─────────────────────────────────────────────────────────────────
DMA 0: TMA→s[0]
        Prod[0]████████████████████████████████████
DMA 1:          TMA→s[1]  ██████████████████████████
Comp 0:          wait[0]→WGMMA→cons[0]  █████████
DMA 2:                    TMA→s[2]      █████████████████████
Comp 1:                    wait[1]→WGMMA→cons[1]   ████████
DMA 3:                              TMA→s[0]       ███████████
Comp 2:                              wait[2]→WGMMA→cons[2]███
DMA 4:                                        TMA→s[1]██████
Comp 3:                                        wait→WGMMA→cons
Comp 4:                                               wait→WGMMA→cons
─────────────────────────────────────────────────────────────────
          DMA warp领先Compute PIPE=3步，预取数据覆盖TMA延迟
```

从timeline可见：(1) DMA warp始终比compute领先3步（PIPE步），使TMA异步拷贝的延迟被连续的计算迭代完全隐藏；(2) 只有在第一个PIPE次迭代开始时compute需等待DMA完成（cold start latency），后续迭代因流水线已充满而无额外等待；(3) prod/cons barriers的arrive/wait支持pipeline slot的循环重用（mod PIPE索引）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- Pipeline深度选择：通常为2-4——太浅则TMA延迟暴露，太深则shared memory不足或寄存器压力过大
- Shared memory组织：为每个pipeline slot分配独立的buffer（如`sA[T_M, T_K, 3]`——第3维是pipeline depth）
- Barrier数量：每pipeline slot需要一对prod/cons barriers，外加一个copyout barrier
- Backwards anti-dependency：producer在写入buffer slot前必须等待consumer完成上一轮使用该slot的计算——这通过cons barrier实现
- Backwards edge在Cypress IR中显式编码为dashed edges in dependence graph（Figure 12），编译器的pipelining transformation自动插入

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
- ThunderKittens: Simple, Fast, and Adorable Kernels

## LCSF (Load-Compute-Store-Finish) Template

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LCSF (Load-Compute-Store-Finish) Template 是 ThunderKittens 提出的统一异步 GPU kernel 编程模型，基于经典生产者-消费者范式。它将 thread block 内的 warp 划分为两种角色——load/store worker（负责 HBM ↔ shared memory 数据搬运）和 compute worker（负责 register/shared memory 内计算）——并将 kernel 分解为四个阶段函数：Load（指定从 HBM 异步加载哪些 tile 到 shared memory pipeline buffer，通过 arrive barrier 通知 compute worker）、Compute（用 tile 操作原语执行 mma/softmax 等计算，完成后通过 arrive 通知 load worker 可覆盖已消费 buffer）、Store（将结果 tile 从 shared memory 异步写回 HBM）、Finish（退出前保存最终状态）。LCSF 是 TK 对 block 级并行性的核心抽象，用户只需填充这四个函数，框架自动管理 multi-stage pipeline buffer、同步 barriers 和 TMA descriptor 创建。对比 FlashAttention-3 的"ping-pong scheduler"（手动管理两个 buffer 轮换），LCSF 用统一的 pipeline buffer 抽象替代，将 attention 实现从 2325 行 CUTLASS 代码缩减到 217 行 TK 代码。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LCSF attention kernel 执行流程（H100, d=64, 2-stage pipeline）：
```
struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 12, INPUT_PIPE_STAGES = 2;

    // Producer: load worker (warp_id==0)
    static void load(args) {
        tma::expect(inputs_arrived, block.k, block.v);
        tma::load_async(block.k, globals.K, {batch, head, iter, 0});
        tma::load_async(block.v, globals.V, {batch, head, iter, 0});
    }
    // Consumer: compute warpgroups (3 warpgroups × 4 warps)
    static void compute(args) {
        warpgroup::mm_ABt(att, q_reg, block.k);     // Q @ K^T via WGMMA
        warpgroup::mma_async_wait();
        sub_row(att, att, max_vec);   // online softmax
        exp(att, att);  div_row(att, att, norm_vec);
        copy(att_bf16, att);           // fp32 → bf16
        warpgroup::mma_AB(o_reg, att_bf16, block.v); // att @ V
        warpgroup::mma_async_wait();
        arrive(inputs_finished);       // 通知 load worker 该 input buffer 可覆盖
    }
    static void store(args)  { tma::store_async 结果 → HBM }
    static void finish(args) { div_row 最终归一化 + store }
};
```
时间线：load worker (warp 0) 通过 TMA 异步加载 K/V tile 到 2-stage buffer slot[0]→arrive→compute 在 slot[0] 上执行 while load worker 预取 slot[1]→compute arrive(inputs_finished) 释放 slot[0]→循环。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 通过 C++ template metaprogramming 实现：(1) 用户定义 LCSF template struct，填充 Load/Compute/Store/Finish 四个静态函数；(2) 设置编译时常量 NUM_CONSUMER_WARPS（compute worker 数量）和 INPUT_PIPE_STAGES（pipeline 深度）；(3) 使用 TK tile 原语（warpgroup::mma_ABt, tma::load_async, arrive）编写逻辑；(4) 调用 kittens::prototype::lcsf::kernel<template> 启动。通过 NUM_CONSUMER_WARPS 调节 occupancy——更多 worker 增加并行度但减少每 worker 的寄存器配额。LCSF 已验证通用性：用于 GEMM (40行)、attention (217行)、long convolution (131行)、Mamba-2 (192行)、linear attention (282-316行)、rotary (101行)、fused layernorm (146行) 等多种 workload，全部匹配或超过 state-of-the-art 性能。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels

## Persistent Grid / Persistent Kernel Launch (GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Persistent Grid（持久化网格）是一种 GPU kernel 调度策略：launch 恰好等于 GPU 物理 SM 数量的 thread block（H100 为 132 个），让每个 block 常驻在其 SM 上，通过 task iteration 循环处理多个 tile 任务。每个 block 完成当前 task 后不退出，而是加载下一个 task 坐标继续执行。ThunderKittens 用 persistent grid：(1) 消除重复 block launch 的 setup/teardown 开销；(2) 利用 load worker 在 finish 阶段预取下一个 task 的数据，消除 pipeline bubble；(3) 配合 block order scheduling 最大化 L2 cache reuse。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// Host: 只 launch 132 个 block（== H100 SM 数量）
dim3 grid(132); // PERISISTENT_GRID=true
// Device 端伪代码:
while (task_id < total_tasks) {
    common_setup: task_id = task_iter * 132 + blockIdx.x;
    // Load→Compute→Store→Finish 流水线处理当前 task
    for (iter over K tiles) { tma::load + warpgroup::mma }
    store output to HBM;
    // 预取下一个 task 的输入 (overlap with current finish)
    task_iter++;
}
```
TK 实验（GEMM, M=N=4096）：persistent vs non-persistent——K=64: 108 vs 93 TFLOPS (+16%); K=128: 184 vs 161 (+14%); K=256: 309 vs 271 (+14%); K=512: 450 vs 414 (+9%); K=1024: 600 vs 565 (+6%)。大 K 时优势递减，因 compute 时间增长使 launch 开销占比减小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 通过 `kittens::prototype::lcf::kernel<template>` 的 `PERISISTENT_GRID` 模板参数实现——true 时 host grid 固定为 132，device 端自动在 common_setup 中用 while 循环 + task_id 映射替代一次性遍历。需配合 block order scheduling 最大化 L2 reuse。局限性：总 task 数少于 SM 数时部分 SM 闲置；极计算密集的 kernel 收益递减。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels

## Swizzled Shared Memory Layout (NVIDIA Hopper)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Swizzled Shared Memory Layout 是通过对 shared memory 地址进行 XOR 变换来重排数据元素到不同 memory bank 的布局技术，消除 bank conflict（多线程同时访问同一 bank 导致串行化）。NVIDIA Hopper shared memory 由 32 个 bank 组成（4 字节宽），bank conflict 显著增加访问延迟。ThunderKittens 提供三种编译时自动选择的 swizzle：32 字节（4-way conflict, width≤16 的 tile）、64 字节（2-way conflict, width≤32）、128 字节（0 conflict, width≤64 且 bf16）。与 row-major（8-way conflict when loading to tensor core layout）和 padded 布局（无 conflict 但地址非对齐，不兼容 TMA/HGMMA）相比，swizzle 在消除 conflict 的同时保持地址对齐，兼容 H100 的 TMA bulk copy 和 WGMMA 指令。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
128 字节 swizzle 实现：
```
bf16* swizzled_layout_128B(bf16 *data, int r, int c) {
    uint64_t addr = (uint64_t)&data[r * columns + c];
    return (bf16*)(addr ^ (((addr % (128*8)) >> 7) << 4));
}
```
原理：取地址对 128×8=1024 字节的余数，右移 7 位，左移 4 位得 XOR 值（16 字节粒度位翻转），与原始地址 XOR——使同一列不同行的连续元素映射到不同 bank。

TK 编译时自动选择：
```
if (tile_width <= 16 && type==bf16)  → 32B swizzle  // 4-way conflict
elif (tile_width <= 32)              → 64B swizzle  // 2-way conflict
else                                  → 128B swizzle // 0 conflict, WGMMA/TMA compatible
```
NCU profiling 证据：FlashAttention-3 (CUTLASS) 存在 9.6-way bank conflict → shared memory stall 0.92 cycles；TK attention kernel 的 shared memory stall 仅 0.14 cycles —— 85% reduction，归功于自动 swizzle 选择。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 中用户无需手动选择 swizzle——定义 `st_bf<H, W>` shared tile 时编译器自动根据 W 和数据类型选择最优布局。自动 swizzle 是 TK 对比 CUTLASS 的核心优势之一：CUTLASS 需要程序员手动管理 shared memory layout（经常导致保留的 bank conflict），TK 将 layout 作为框架内部优化自动化。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels

## Block Order Scheduling / L2 Cache Reuse (GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Order Scheduling 是通过控制 GPU thread block 在 grid 中的执行顺序来最大化 L2 cache reuse 的调度策略。GPU 的 thread block 无法直接通信——数据共享必须通过 HBM（经过 L2 cache）。当相邻执行的 block 访问相同数据区域时，数据保持在 L2 cache（50MB, 12 TB/s）而非从 HBM（3 TB/s）重载。Block order 由 GPU 硬件调度器根据 blockIdx 分配决定，但 kernel 可通过 grid dimension 和 blockIdx→数据坐标的映射影响哪些 block 连续执行。TK 展示了 block order 对性能的惊人影响。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GEMM M=N=K=16384 的对比：
```
策略 A: 3D Stride {8, N, M/8}
  row = 8*(task_id/super_repeat) + task_id%8
  col = (task_id/8) % N
  → 相邻 block 在 row 方向连续 → A 矩阵行数据 L2 reuse
  → HBM: 982 GB/s, 805 TFLOPS

策略 B: Row-Major {N, M}
  row = blockIdx.x / N, col = blockIdx.x % N
  → 相邻 block 遍历不同 row，cache 无法保存 B 矩阵所有列
  → HBM: 3070 GB/s (L2 miss!), 仅 392 TFLOPS
```
策略 B 的 HBM 带宽更高但性能减半——高带宽意味着 L2 cache miss，数据被迫从慢速 HBM 加载。策略 A 通过连续 block 复用相邻数据使大部分访问命中 L2。

Attention forward (d=128):
- 优化 order {N, H, B}（sequence 连续）: HBM 213 GB/s, 600 TFLOPS
- Naive order {B, H, N}（batch 连续）: HBM 2390 GB/s, 494 TFLOPS
优化 block order 提升 21% 性能，同时 HBM 带宽降低 91%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TK 在 common_setup 中通过 task_id→(row,col) 映射实现——用户选择 3D stride 维度（如 GEMM 用 SUPER_M 参数控制 cluster 行数），映射公式直接影响 L2 reuse。与 persistent grid 配合效果最佳：block 连续执行多个 task 时，L2 cache 可以在 task 之间保持数据。

涉及论文标题：
- ThunderKittens: Simple, Fast, and Adorable Kernels

## Register Tensor Reinterpretation / View Instruction (寄存器张量重解释)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Register Tensor Reinterpretation（View指令）是Tilus VM中的零开销操作，允许在不进行任何数据移动或拷贝的情况下，同时改变register tensor的数据类型（dtype）和layout。其正确性条件：两个tensor分布在同一数量的线程上（32 threads），且每个线程持有的总bit数相同。例如，32个线程各持有24 bits，既可解释为3×uint8（3×8 bits），也可解释为4×int6（4×6 bits），通过View指令在registers内即时切换（Figure 2c）。View是Tilus消除Triton shared memory layout conversion瓶颈的核心机制：Triton必须通过shared memory中转来改变register tensor layout，而Tilus直接在registers内完成。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
在低精度weight loading pipeline中View指令的使用：
```
# b_tile在32 threads的registers中，每线程持有 3×u8 = 24 bits
b_tile = LoadShared(shared_tile, dtype=u8, layout=local(3).spatial(32), offset=0)

# View: 零开销将24 bits/thread reinterpret为 4×int6，同时改变layout
# 原layout: local(3).spatial(32) —— 每线程3个连续u8
# 新layout: local(2,1).column_spatial(4,8).local(2,1) —— Tensor Core兼容layout
# 不产生任何PTX指令（纯编译器metadata操作）
b_tile = View(b_tile, dtype=i6, layout=local(2,1).column_spatial(4,8).local(2,1))
```

对比Triton（Figure 1a）weight loading pipeline中的Step 4（shared memory layout conversion）和Ladder（Figure 1b）的Step 3-4（shared memory中转），View指令将两者都消除为registers内的零开销操作。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
View指令的实现基于Tilus的代数layout系统。编译器在IR层面维护每个register tensor的dtype+layout属性，View指令仅修改这些metadata属性——不生成任何实际的PTX指令或数据移动。在代码生成阶段，后续指令（如Cast、Dot）根据更新后的dtype+layout信息生成正确的per-thread操作序列。开发者使用方式：`b = View(a, [dtype], [layout])`，其中dtype和layout参数至少提供一个。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## Vectorized Casting with PRMT and LOP3 (向量化类型转换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vectorized Casting是Tilus将低精度register tensor高效转换为标准浮点类型（如float16）的机制，使用CUDA PTX的PRMT（Permute Bytes）和LOP3（三输入逻辑操作）指令在registers内完成，无需shared memory或inter-thread通信。PRMT指令从两个32-bit源寄存器（视为8个连续字节）中按4-bit selector抽取并重新排列4个输出字节；LOP3指令对三个32-bit输入执行任意布尔逻辑操作（通过8-bit truth table编码）。结合bitwise AND/SHIFT/OR指令，这些操作可在registers内实现高效的向量化dequantization（如int6→float16）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
低精度INT6→FP16的寄存器内向量化casting流程（单线程视角）：
```
# 输入：32-bit寄存器 r 包含4个packed int6值（bits布局: [6:0, 14:8, 22:16, 30:24]）
# 输出：4×FP16 values (每个占16 bits)

# Step 1: 提取各int6值并符号扩展到8-bit
# 使用PRMT配合selector使各byte独立提取；或使用SHIFT+AND提取4个值
v0 = (r >> 0)  & 0x3F;  v0 = v0 << 26 >> 26  # sign extend int6 to int32
v1 = (r >> 8)  & 0x3F;  v1 = v1 << 26 >> 26
v2 = (r >> 16) & 0x3F;  v2 = v2 << 26 >> 26
v3 = (r >> 24) & 0x3F;  v3 = v3 << 26 >> 26

# Step 2: INT32 → FP16 conversion
# (使用CUDA intrinsic __float2half或PTX cvt指令)
f0 = int2float(v0) * dequant_scale  # dequantize
f1 = int2float(v1) * dequant_scale
f2 = int2float(v2) * dequant_scale
f3 = int2float(v3) * dequant_scale

# PRMT优化版：一次指令处理2个32-bit寄存器对的4字节重排
# 可加速Step 1的byte extraction和rearrangement
```

Tilus编译器在每个低精度类型的Cast指令code emitting时自动选择最优的PRMT/LOP3/bitwise指令序列，实现vectorized casting。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Tilus的编译流程Step 3（低精度类型降低）中，Cast操作被展开为针对目标硬件（CUDA）的指令序列。开发者只需写`b = Cast(a, dtype=f16)`，编译器自动根据源dtype选择PRMT selector pattern和转换逻辑。在CUDA C中，PRMT通过`__byte_perm(a, b, selector)` intrinsic使用（selector=0x7777时编译器自动mask sign-extend bit），需要sign extension模式时则用inline PTX assembly。LOP3通过`__lop3(a, b, c, truth_table)`或PTX inline assembly访问。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## Software Pipelining in Low-Precision GPU Kernels (低精度GPU Kernel软件流水线)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Software pipelining（软件流水线）在低精度GPU kernel中指将全局内存→共享内存的数据拷贝与Tensor Core计算重叠执行的优化技术。Tilus通过CopyAsync、CopyAsyncCommitGroup()、CopyAsyncWaitGroup(n)三个VM指令实现对异步拷贝流水线的声明式控制。CopyAsync发起一次async copy任务（触发cp.async硬件指令），CommitGroup标记一组任务的边界，WaitGroup(n)阻塞直到in-flight group数≤n。配合shared memory的多级缓冲，可实现global→shared copy与Tensor Core computation的完整重叠——当Compute正在处理tile_i时，DMA已在加载tile_{i+1}到shared memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。
双缓冲软件流水线伪代码（decode stage matmul, K维循环）：
```
# 初始化: prefetch第一K tile
CopyAsync(shared_buf[0], global_weight_tile[0])
CopyAsyncCommitGroup()

for k in range(1, num_k_tiles):
    CopyAsync(shared_buf[k%2], global_weight_tile[k])        # 异步加载下一tile
    CopyAsyncCommitGroup()
    CopyAsyncWaitGroup(2)                                      # 最多2组in-flight

    a_tile = LoadGlobal(A_global, layout, offset=[:, k-1:])   # 从global加载activation
    b_tile = LoadShared(shared_buf[(k-1)%2], layout, offset)  # 从shared加载权重
    b_tile = View(b_tile, target_dtype, target_layout)         # 零开销reinterpret
    b_tile = Cast(b_tile, f16)                                  # 向量化casting
    C_accum = Dot(a_tile, b_tile, C_accum)                     # Tensor Core计算
    Synchronize()                                               # 等待当前iteration完成

# 处理最后一个tile
CopyAsyncWaitGroup(0)
# ... (最后一个tile的计算)
```

与Triton（Figure 1a）对比：Triton同样支持cp.async但受限于shared memory layout conversion瓶颈；与Ladder（Figure 1b）对比：Ladder的primitive-style scheduling根本不支持software pipelining（weight loading与computation完全串行）。Tilus结合了pipelining和零开销layout reinterpretation两个优势。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Tilus的CopyAsync在编译时映射为PTX cp.async指令（或cp.async.v4向量化版本），CommitGroup/WaitGroup映射为cp.async.commit_group和cp.async.wait_group。编译器自动选择向量化宽度并计算shared memory buffer大小。软件流水线在batch>1的decode场景中尤为关键——此时compute load增大，pipelining的overlap收益显著。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

## SpGEMV (Sparse GEMV) with Quantized Key Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SpGEMV (Sparse Generalized Matrix-Vector multiplication) with Quantized Key Cache 是 Twilight 提出的高效 attention weight 估计 kernel。核心设计：对 K cache 做 per-head asymmetric INT4 quantization，在 GPU 上执行 q_fp16 @ K_int4 的稀疏矩阵向量乘法来估计 attention weights，同时将 memory access 降至 FP16 K cache 的 1/4。这是 top-p sparse attention 的关键性能 enabling 技术——top-p 比 top-k 需要更高的数值精度（不仅序数正确，还需数值准确性），因此不能使用极低精度（1-2 bit），但 FP16 又浪费带宽。实验证明 4-bit 是最优 sweet spot。

从kernel调度角度拆解术语，给出具体例子。
```
// INT4 K Cache SpGEMV kernel (基于FlashInfer decode attention kernel修改)
Input: q ∈ R^{BS×H×d}, K_int4 ∈ R^{N×d/2} (paged, per-head dynamic quantized)
Output: W_approx ∈ R^{BS×H×N}  (estimated attention weights)

Per thread block:
  // 2-stage software pipeline
  for k_iter in 0..d/Kt:
    // Stage 1: async load + dequantize current tile
    cp.async: GMEM[K_int4[k_iter]] → SMEM[buf_ping]   // load INT4 K tile
    cp.async.commit_group
    cp.async.wait_group
    
    // Dequantize in shared memory:
    // K_fp16 = (K_int4_unpacked - zero) * scale
    // Use per-head dynamic scale/zero stored in paged layout
    // INT4→FP16 conversion via PTX asm (FasterTransformer-style)
    K_tile_fp16 = dequantize_int4_to_fp16(SMEM[buf_ping])
    
    // Stage 2: dot product (overlapped with next tile async load)
    for i in range(BS×H):
      W_approx[i, :] += dot(q[i, k_iter:k_iter+Kt], K_tile_fp16)
    
    // swap ping/pong buffers
    
  return W_approx
```

Bit-packing: 两个 INT4 元素打包为 uint8_t（interleaved packing），地址计算 remap 到 4-bit granularity（halving effective byte offset）。Dequantization 前先加 offset +128 转 unsigned 再 pack。

术语一般如何实现？如何使用？
基于 FlashInfer 的 attention decoding kernel 修改，自研 CUDA/Triton kernel。INT4 K cache 使用 paged layout 与 FP16 KV cache 对齐。Per-head 动态量化参数（FP16 scale + zero）同样使用 paged 布局。额外内存开销：1/8 FP16 KV cache。该 kernel 也可复用于其他需要估计 attention weights 的 sparse attention 方法。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

## Top-p via Binary Search (GPU Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-p via Binary Search 是 Twilight 在 GPU 上高效实现 top-p attention weight 选择的并行算法。直接的 top-p 实现需要降序排序后累积——在 GPU 上极为低效（O(N log N) sorting）。该算法采用 parallel-friendly binary search：在 [0, max(W)] 区间二分搜索阈值，每次迭代通过 tensorized element-wise 操作计算累积概率并比较，避免排序且不物化中间变量。收敛到精度 ε 需 O(log(range/ε)) 次迭代（typically 8-12 次）。

从kernel调度角度拆解术语，给出具体例子。
```
// Algorithm 1: Top-p via Binary Search (GPU kernel, fully tensorized)
Input: W ∈ R^{BS×H×N} (normalized attention weights), threshold p, tolerance ε
Output: I (selected indices), M ∈ {0,1}^{BS×H×N} (mask)

l = 0, r = max(W)  // 所有element-wise操作融合为单次循环

repeat:
  m = (l + r) / 2
  
  // Fused operations (tensorized on GPU, single pass):
  // ① where(W < m, 0, W) — mask below threshold
  // ② sum(masked_W)      — compute cumulative probability
  masked = where(W >= m, W, 0.0)
  cumsum = sum(masked)
  
  if cumsum >= p: l = m  // cumulative enough → raise threshold, prune more
  else: r = m            // not enough → lower threshold, keep more
  
until max(W[W > r]) - min(W[W >= l]) < ε

M = (W >= l)  // final mask
I = indices(M == 1)
return I, M
```

关键优化：(a) element-wise max/where/sum 融合为单次 register-level 循环，不物化中间变量（如 W0）；(b) 8-12 次迭代即可收敛（ε=0.01）；(c) 比 sorting-based 方法快 O(N) 倍。

术语一般如何实现？如何使用？
修改 FlashInfer 的 top-p sampling kernel（原用于 LLM text generation token sampling），应用场合从 "选下一个 token" 变为 "选 attention weights"。使用 CUDA thread block 并行化，每 thread 处理一部分 token 的 element-wise 操作。适用于所有需要 top-p selection 的 GPU kernel。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

## Head-wise Varlen Attention with GQA Support

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Head-wise Varlen Attention 是支持每个 attention head 使用不同 sequence length（不同 budget）的 sparse attention kernel。由于 top-p 使不同 head 选择不同数量的 token（B1 因 head 而异），传统 padded attention（将所有 head pad 到 max B1）浪费计算；head-wise varlen attention 为每 head 分配恰好 B1(head) 个 token 的计算，消除 padding waste。对 GQA（Group Query Attention），升级为 group-wise varlen：同一 query group 内取各 head 选择 token 的 union，以 group 粒度做 varlen attention——平衡了实现效率与兼容性。

从kernel调度角度拆解术语，给出具体例子。
```
// MHA: head-wise varlen
// 不同 head 选不同 B1(head) token → 直接 varlen attention per head
// FlashInfer load balancing: flatten head dim, 按 B1 per head 分配 compute

// GQA: group-wise varlen (Twilight, Appendix B.2)
// e.g., LLaMA-3.1: 4 Q heads → 1 KV head per group
for each query_group in 1..H_kv:
    I_group = empty_set()
    for each q_head in query_group:
        I_group = I_group ∪ I_head  // union of selected tokens
    B_group = |I_group|
    // Attention: Q_group ∈ R^{4×d} @ K[I_group]^T → group_size × B_group
    // 所有 heads in group 共享 K/V[I_group]

// 方案对比 (Figure 13):
// Padded: 所有 head pad 到 max_budget → 大量浪费
// Head Varlen: 每 head 独立加载 K/V → 重复加载（GQA中多head共享同一KV）
// Group Varlen: union per group → 平衡计算浪费与重复加载
```

术语一般如何实现？如何使用？
基于 FlashInfer 的 varlen attention + load balancing。Flattened paged KV cache layout，通过 per-group 的 B_group 做 scatter-arrange。GQA group union 使同一 group 内共享 KV 的 heads 不产生重复加载。适用于所有需要动态 per-head budget 的 sparse attention 方法。

涉及论文标题：
- Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning


---

## Tile-Level Operator Fusion (WELDER)

术语是什么？
Tile-Level Operator Fusion是WELDER提出的一种统一融合框架，将DNN算子的融合从传统的"规则匹配（rule-based matching）"抽象为"tile-graph连接调度"。核心理念是：不限制哪些operator类型可以融合，而是通过SetConnect在tile-graph的edge上指定数据复用的memory level，Propagate自动对齐tile shape，然后由traffic cost model判断融合是否有益（减少total traffic）。如果两个reduction-based operator（如Matmul+Softmax）在shared memory level连接后，traffic减少了，WELDER就自动生成fused kernel，无需预定义的融合规则。

与Ansor/TVM的register-level element-wise融合和AStitch的shared-memory rule-based融合的本质区别：WELDER不预先限制可融合的operator类型——任何能用tensor expression描述的operator都可以参与tile-level fusion。这使WELDER自动发现89种含两个以上reduction operator的非常规融合模式，最大将48个operator融合为单个kernel。

从kernel调度角度拆解术语：
Tile-Level Operator Fusion的执行流程：

```
输入: DNN Graph G = {Conv, ReLU, MaxPool, ...}

For each edge (op_i → op_j) in topological order:
  For each memory level L in [Register, SharedMem, GlobalMem]:
    SetConnect(edge, L)            // 尝试在L层连接
    
    subgraph = ExtractSubgraph(op_i, level=0)
    // 提取所有connect_level > 0的连续边形成的连通子图
    
    // 对每个subgraph搜索最优tile配置
    for subtile in EnumerateSubtiles(subgraph):
      config = Propagate(subgraph, subtile)  // 推断所有tile shape
      if MemFootprint(subgraph) > L.capacity: continue
      configs.push(config, priority=MemTraffic(subgraph))
    
    top_configs = TopK(configs, k)
    for config in top_configs:
      // 递归上层sub-graph调度
      for node in subgraph.nodes:
        upper = ExtractSubgraph(node, level+1)
        SubGraphTiling(upper, level+1, config)
    
    latency = Min(Profile(top_configs))
    if latency < best_latency:
      best_level = L

  SetConnect(edge, best_level)  // 选择最优连接层
```

融合示例（BERT attention: Matmul Q*K + Softmax）:
- Ansor: 无法融合（两个都是reduction-based operator，无匹配规则）
- WELDER: SetConnect(Matmul→Softmax, SharedMem) → Propagate → MemTraffic评估 → 发现[16×128] tile节省69% traffic → 自动生成fused kernel

融合示例（NAFNet: pointwise Conv + Norm + pointwise Conv）:
- TensorRT: 无对应fusion rule
- WELDER: 在shared memory连接 → 自动fuse，3.09× speedup

术语一般如何实现？如何使用？
WELDER将融合问题转化为图优化问题。在code generation阶段，shared memory level的连接通过Load/Store Rewriting实现：原独立kernel的global memory load/store被改写为shared memory access，然后组合为一个fused kernel。register level的连接通过TVM compute_inline实现。用户仅需提供ONNX graph，WELDER自动完成融合决策和codegen。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

## Shared Memory Tile Connection (WELDER)

术语是什么？
Shared Memory Tile Connection是WELDER中tile-graph在GPU shared memory（L1）层的连接机制：通过SetConnect(edge, SharedMem)将两个相邻operator-tile通过shared memory中的reuse-tile直接连接，使第一个operator-tile的输出数据tile留在shared memory中，被第二个operator-tile直接消费，无需经过DRAM往返。这是WELDER fusion能力的关键——传统的Ansor/TVM仅支持register级的element-wise融合（如Conv+ReLU），而shared memory级的连接可以处理更复杂的operator组合（如Matmul+Softmax、Conv+Pool等reduction-based operator pair）。

从kernel调度角度拆解术语：
Shared Memory Tile Connection的kernel执行过程：

```
GPU Kernel: Fused Matmul + Softmax (shared memory level connection)

// 全局参数
grid_dim = (ceil(M/BM), ceil(N/BN))
block_dim = (128 threads)  // GCD of both operator tile thread counts

For each thread block (bm, bn):
  // Step 1: Load input tiles from DRAM → shared memory
  LoadTiles:
    A_tile[BM×BK] ← DRAM[A_addr + bm*BM×BK]  (coalesced 128B transactions)
    B_tile[BK×BN] ← DRAM[B_addr + bn*BN×BK]  (coalesced 128B transactions)
  
  // Step 2: Matmul operator-tile execution
  ComputeTile (Matmul):
    for kk in 0..BK step 16:  // TensorCore fragment size
      A_frag[16×16] ← shared_mem[A_tile[kk:kk+16]]  // ldmatrix
      B_frag[16×16] ← shared_mem[B_tile[kk:kk+16]]  // ldmatrix
      C_accum[16×16] += A_frag × B_frag  // mma.sync (TensorCore)
    
    // C_accum written to shared_mem[C_tile[BM×BN]]
    // ← KEY: C_tile stays in shared memory, NOT written to DRAM
  
  // Step 3: Inter-operator data reuse in shared memory
  // Softmax reads C_tile directly from shared memory
  __syncthreads()  // ensure Matmul writes visible
  
  // Step 4: Softmax operator-tile execution
  ComputeTile (Softmax):
    for row in range(BM):
      // All threads cooperatively process row
      row_data = shared_mem[C_tile[row, 0:BN]]
      max_val = warp_reduce_max(row_data)
      exp_vals = __expf(row_data - max_val)
      sum_exp = warp_reduce_sum(exp_vals)
      D_tile[row, 0:BN] = exp_vals / sum_exp
  
  // Step 5: Write final result to DRAM
  StoreTiles:
    DRAM[D_addr + ...] ← D_tile[BM×BN]  (coalesced 128B transactions)
```

Shared memory management:
```
// Buffer allocation (bestfit):
// - A_tile: BM×BK×4 bytes (FP32)
// - B_tile: BK×BN×4 bytes
// - C_tile/D_tile: BM×BN×4 bytes (reuse-tile, shared between Matmul output and Softmax input)
// - Padding: align to 32B for TensorCore, avoid bank conflicts

// Liveness analysis + bestfit offset calculation:
Allocation order: A_tile, B_tile, C_tile, (free A_tile after Matmul), ...
Finally: shared_mem[D_tile] → DRAM
```

术语一般如何实现？如何使用？
实现细节：(1) Load/Store Rewriting——TVM TIR pass将独立kernel的global memory access改写为shared memory access，添加memory fences防race condition；(2) Shared Memory Management——liveness analysis + bestfit算法统一管理所有shared memory buffer，考虑32B alignment；(3) Block/threadIdx remapping——Transpose等算子需blockIdx映射，2D thread block通过remapping与1D thread block并存；(4) Block size alignment——所有operator-tile的线程数取GCD作为统一block size（≥128 warp scheduler requirement且≤1024 max）。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

## Hardware-Aligned Tile Search (WELDER)

术语是什么？
Hardware-Aligned Tile Search是WELDER在枚举tile shape时施加硬件约束的penalty机制。在SubGraphTiling中，EnumerateSubtiles从size=1开始扩展tile shape（类似Roller的tile shape expanding approach），但WELDER在MemTraffic cost model基础上附加三个hardware-aware penalty：(1) **Uncoalesced Memory Access Penalty**——若tile shape导致非连续128B transaction的global memory access，按实际需要的transaction数计算额外traffic；(2) **Parallelism Penalty**——若tile shape太大导致硬件并行度不足（如V100上tile数 < 128），按core utilization比例增加traffic；(3) **Capacity Penalty**——若MemFootprint > target memory capacity，施加infinite penalty直接淘汰。

从kernel调度角度拆解术语：
EnumerateSubtiles with hardware penalties：

```
EnumerateSubtiles(graph, last_config):
  // 从最小的合法tile shape开始
  init_tile = {axis: 1 for axis in output_axes}
  
  for axis in output_axes:
    tile = init_tile.copy()
    while tile[axis] < tensor_dim[axis]:
      // 扩展当前轴的tile size
      tile[axis] = expand_toward_hardware_alignment(tile[axis])
      
      // Penalty 1: Uncoalesced access check
      // CUDA GPU: 128B per transaction (32 FP32 elements)
      if not is_coalesced(tile, transaction_width=128B):
        extra_traffic = calculate_extra_transactions(tile)
      
      // Penalty 2: Parallelism check
      // V100: 80 SMs × 4 warp schedulers × 32 threads = 128 min parallelism
      num_parallel_tiles = total_elements / tile_size
      if num_parallel_tiles < hardware_parallelism:
        extra_traffic *= (hardware_parallelism / num_parallel_tiles)
      
      // Penalty 3: Capacity check
      footprint = MemFootprint(graph_with_tile_config)
      if footprint > target_memory_capacity:
        continue  // skip, infinite penalty
      
      adjusted_traffic = MemTraffic(graph) + extra_traffic
      configs.push(tile_config, priority=adjusted_traffic)
    
  // TensorCore constraints
  for axis marked as MMA_axis:
    // M, N, K must be multiples of MMA fragment size (e.g., 16 for FP16)
    enforce_tile[axis] % MMA_fragment_size == 0
  
  return configs  // sorted by adjusted_traffic ascending
```

V100硬件参数：
- Transaction width: 128B (32 × FP32 elements)
- Min parallelism: 128 (80 SMs × budget for multi-SM occupancy)
- Max block size: 1024 threads
- MMA fragment: 16×16×16 (FP16 TensorCore mma.m16n8k16)

术语一般如何实现？如何使用？
在WELDER的SubGraphTiling中，penalty直接加到MemTraffic上作为优先队列的排序键。这确保搜索首先探索硬件友好的tile shape，而非纯traffic最小但硬件不友好的配置。对于TensorCore，额外添加MMA fragment size整除约束。对于Block size，取所有operator的线程数GCD，若<128则设为128，若>1024则取1024。

涉及论文标题：
- Welder Scheduling Deep Learning Memory Access via Tile-graph

---

## GPU Memory Pool

术语是什么？
GPU Memory Pool（GPU 内存池）是一种预分配 GPU 内存的管理机制，用于减少运行时频繁 cudaMalloc/cudaFree 系统调用的开销。在 HuntKTm 中，task scheduler 在 task 首次内存分配前根据预测的 memory footprint 预分配一个 memory pool。后续所有 allocation 请求若可从 pool 满足，则直接从 pool 返回预分配内存（无需 OS 级别的 GPU memory allocation 系统调用）；deallocation 请求不真正释放内存，而是将内存保留在 pool 中供后续 reuse。pool 中的内存仅在 application 退出时完全释放。

从 kernel 调度角度拆解术语：
GPU memory pool 在 HuntKTm 任务执行中的运转流程：

```
Task 生命周期中的 Memory Pool:

1. Task Scheduler 初始化阶段（runtime，dispatch 后）:
   predicted_footprint ← lazy engine 汇总的 memory requirement
   cudaDeviceGetDefaultMemPool(&pool)  // 获取 CUDA 默认 memory pool
   cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, predicted_footprint)
   // 设置 release threshold：memory usage 低于此 threshold 时不释放
   // 确保 pool 中的内存不会被 CUDA runtime 自动回收

2. Task 执行阶段（lazy engine 顺序执行 deferred operations）:
   cudaMallocAsync(&ptr, size, stream):
     if pool.free_memory >= size:
       直接从 pool 返回预分配内存 → 避免系统调用
     else:
       触发真正的 GPU memory allocation

   cudaFreeAsync(ptr, stream):
     将内存归还 pool → 不真正释放
     // 即使 task 内部多次 alloc/free，pool 保持足够内存

3. Task 退出:
   pool 内所有内存被释放
```

对 kernel 调度的关键影响：
- 减少运行时 memory alloc/free 开销：HuntKTm vs HuntKT（无 memory pool）在单 task 执行中提升 speedup（M1: 3.27×, M2: 3.17× vs Serial）
- 与 memory manager（编译期 liveness analysis）协同：memory manager 减少 peak memory → pool 的 predicted_footprint 更小 → 更多 task 可同时运行
- 消除频繁系统调用导致的 kernel launch 延迟

术语一般如何实现？如何使用？
通过 CUDA 12.0+ 的 Stream-Ordered Memory Allocator API：`cudaMallocAsync` 和 `cudaFreeAsync` 使用设备默认 memory pool。`cudaDeviceGetDefaultMemPool` 获取 pool handle，`cudaMemPoolSetAttribute` 设置 `cudaMemPoolAttrReleaseThreshold` 控制内存释放阈值。HuntKTm 的 lazy engine 将所有 memory allocation 转化为 cudaMallocAsync，并在 task 退出时通过 pool 的 release threshold 防止内存过早归还 OS。此机制与 PyTorch 的 caching allocator 类似，但 HuntKTm 将其集成到通用 GPU 程序的编译-运行时 pipeline 中。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

## Tile-Based Communication (Multi-GPU)

术语是什么？
Tile-Based Communication是Iris提出的多GPU通信编程模型，将跨GPU数据传输操作对齐到与Triton tile计算模型相同的粒度（BLOCK_SIZE_M × BLOCK_SIZE_N）。与传统的bulk-synchronous通信（整个tensor完成计算后才开始跨GPU传输）不同，tile-based communication允许每个tile产出后立即通信——例如在GEMM kernel的main loop中，每个K-iteration产出一个BLOCK_M×BLOCK_N的C_tile后，立刻通过iris.store将其scatter到所有remote GPU。通信原语（load/store/get/put/copy/atomic_*）均操作于tile粒度，与Triton的tl.load/tl.store/tl.dot处于同一语义空间和同一kernel内。

从kernel调度角度拆解术语：
Tile-Based Communication在Fused Sequential GEMM+All-Scatter中的执行流程：
```
@triton.jit
def fused_gemm_all_scatter(A, B, C, heap_bases, ...):
    pid = tl.program_id(0)
    total_tiles = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
    for tile_id in range(pid, total_tiles, NUM_SMS):
        # Compute Phase (standard GEMM loop)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(A + offsets_a)
            b = tl.load(B + offsets_b)
            acc += tl.dot(a, b)  # Tensor Core MMA
        # Communication Phase (same kernel, immediately after tile compute)
        c = acc.to(C.dtype.element_ty)
        for remote_rank in range(world_size):
            iris.store(C + offset, c, cur_rank, remote_rank, heap_bases, mask=mask)
```
关键特征：(1) 同一kernel内无launch/teardown overhead；(2) 通信单位为tile而非整个tensor；(3) 值语义直接从register写remote GPU memory；(4) Triton编译器看到通信操作，可联合调度计算与通信。

术语一般如何实现？如何使用？
Iris提供两类device-side API——值语义(load/store从register到remote memory)和指针语义(get/put/copy做buffer间拷贝)。所有API均需heap_bases参数做指针翻译，翻译开销近乎为零(heap_bases 64 bytes常驻L1 cache)。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton

## Compute-Communication Fused Kernel (Overlap Taxonomy)

术语是什么？
Compute-Communication Fused Kernel是将计算（如GEMM）和通信（如All-Scatter/All-Gather）融合到单个GPU kernel中执行的技术。Iris定义了两大类overlap策略：(1) Unfused——计算和通信在不同kernel中执行，通过CUDA stream并发或CU分区实现overlap；(2) Fused——计算和通信在同一个kernel中交织，通过workgroup specialization或sequential ordering实现。各策略在实现复杂度、资源利用和性能之间提供不同trade-off。

从kernel调度角度拆解术语：
四种overlap pattern对比（GEMM+All-Scatter）：

**Unfused Bulk-Synchronous**: 两个独立kernel顺序执行，中间global barrier。GEMM全完成后通信才开始——GPU资源交替闲置，存在execution bubble。

**Unfused Producer-Consumer**: 两个kernel在不同stream上并发。GEMM kernel使用256 CU，通信kernel使用48 CU。Producer通过atomic_cas(release)通知tile就绪，consumer通过atomic_cas(acquire) spin-lock等待。避免全局barrier但需手动CU分区。

**Fused Sequential**: 单kernel内GEMM tile产出后立即iris.store scatter。最简单的fused模式，仅需几行代码改动。但GEMM和通信在同一workgroup内顺序执行（先算后传），tail latency会增加。

**Fused Workgroup Specialization**: 单kernel内通过pid划分worker角色——256个GEMM workgroup做计算（完成后atomic_cas(release)发信号），48个COMM workgroup spin-lock等信号后iris.put。GEMM和通信在不同CU上并发执行，通信可完全隐藏在GEMM后面（尤其是小N大K场景，N/8后每个tile通信量极小）。代价：资源分配受GEMM（资源密集型）约束。

术语一般如何实现？如何使用？
开发者根据workload特性选择pattern：通信占比小选Fused Sequential；大K小N选Workgroup Specialization（最高效）；需避免worst-case resource allocation选Unfused Producer-Consumer。所有pattern通过Iris device-side API在Triton kernel中实现。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton

## Workgroup Specialization for Multi-GPU Communication

术语是什么？
Workgroup Specialization是将GPU的compute units在单个persistent kernel内划分为不同角色——部分workgroup专门执行计算（GEMM），另一部分专门执行通信（跨GPU数据传输）——通过atomic-based spin-lock传递tile级就绪信号。这是Iris实现最高效compute-communication overlap的kernel调度技术。在8×MI300X（每GPU 304 CU）上，典型配置：256 GEMM workers + 48 COMM workers。

从kernel调度角度拆解术语：
```
@triton.jit()
def wg_specialized_gemm_all_scatter(A, B, C, locks, GEMM_SMS, COMM_SMS, ...):
    pid = tl.program_id(0)
    if pid < GEMM_SMS:
        # === GEMM Worker (前256个workgroup) ===
        for tile_id in range(pid, total_tiles, GEMM_SMS):
            c = gemm_loop(A, B, C)
            tl.store(C + offset, c, mask=mask, cache_modifier=".wt")
            tl.atomic_cas(locks + tile_id, 0, 1, sem="release", scope="gpu")
    else:
        # === COMM Worker (后48个workgroup) ===
        pid = pid - GEMM_SMS
        for tile_id in range(pid, total_tiles, COMM_SMS):
            while tl.atomic_cas(locks + tile_id, 1, 0, sem="acquire", scope="gpu") == 0:
                pass  # spin-lock等待GEMM worker完成
            for remote_rank in range(world_size):
                if remote_rank != cur_rank:
                    iris.put(C + offset, C + offset, cur_rank, remote_rank, heap_bases, mask=mask)
```
GEMM worker使用write-through cache modifier确保COMM worker通过Infinity Fabric coherence看到最新数据。通信tile_n隐藏于GEMM tile_{n+1}的计算期间。

与NVIDIA warp specialization的区别：warp specialization在同一SM/warpgroup内划分角色，workgroup specialization在不同CU上划分角色——后者更适合AMD架构（因AMD静态寄存器分配限制不支持warp specialization直接移植）。

术语一般如何实现？如何使用？
通过triton.jit内pid范围判断实现，开发者需手动实验确定最优GEMM/COMM workgroup数量分配。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton

## Pointer Translation for Multi-GPU Remote Memory Access

术语是什么？
Pointer Translation是Iris实现跨GPU transparent memory access的核心算法——利用symmetric heap的同构性（所有GPU的heap在相同偏移处存储对称数据），将本地指针转换为目标GPU上的等价虚拟地址。翻译步骤：(1) 从heap_bases数组加载源/目标基址 → (2) `offset = ptr - from_base` → (3) `remote_ptr = to_base + offset` → (4) cast回指针类型。heap_bases数组（8 GPU × 8 bytes = 64 bytes）在kernel执行期间常驻L1 cache，翻译开销实测为零（被通信延迟完全主导）。

从kernel调度角度拆解术语：
```
@triton.jit
def __translate(ptr, from_rank, to_rank, heap_bases):
    from_base = tl.load(heap_bases + from_rank)     # GPU_src heap base
    to_base = tl.load(heap_bases + to_rank)          # GPU_dst heap base
    ptr_int = tl.cast(ptr, tl.uint64)
    offset = ptr_int - from_base                     # 同构heap → 相同offset
    to_base_byte = tl.cast(to_base, tl.pointer_type(tl.int8))
    translated_ptr = to_base_byte + offset
    return tl.cast(translated_ptr, ptr.dtype)

@triton.jit
def load(pointer, to_rank, from_rank, heap_bases, mask=None):
    translated_ptr = __translate(pointer, to_rank, from_rank, heap_bases)
    return tl.load(translated_ptr, mask=mask)  # 直接跨GPU load
```
翻译仅需~5条指令（两次tl.load + 减法 + cast），所有Iris device-side API调用前自动执行。Triton编译器可优化翻译指令与后续remote操作之间的pipeline。

术语一般如何实现？如何使用？
开发者无需手动调用__translate——Iris的load/store/get/put/copy/atomic_* API在内部自动完成翻译。仅需传入heap_bases作为所有device-side API的必需参数。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton

## Software Pipeline (T.Pipelined)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Software Pipeline（软件流水线）是 TileLang 中的 T.Pipelined 调度原语，用于将循环体内的数据搬移（如 global→shared memory copy）与计算（如 GEMM/attention）重叠执行，通过异步硬件机制隐藏内存延迟。TileLang 的 Pipeline 机制自动分析循环体内语句的 buffer 使用依赖关系，生成结构化的 interleaved schedule：将前后相邻 iteration 的 Copy 和 GEMM 交错执行（Copy(i+1) 与 GEMM(i) 重叠）。在不同 GPU 架构上自动选择最优硬件路径：(1) Ampere (A100) — cp.async 异步 global→shared copy + cp.async.commit_group + cp.async.wait_group；(2) Hopper (H100) — TMA（Tensor Memory Accelerator）硬件单元 + wgmma.mma_async + warp specialization + mbarrier 同步；(3) AMD CDNA — s_waitcnt lgkmcnt + buffer_load_dword lds 指令。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

T.Pipelined 在不同架构上的调度伪代码：

```
// === Ampere (A100) Pipeline (cp.async) ===
// T.Pipelined(K // block_K, num_stages=2) 自动展开为:

// Prologue: 预取前 2 个 stage
cp.async(A[0], A_shared[0])       // stage 0 load A
cp.async(B[0], B_shared[0])       // stage 0 load B
cp.async.commit_group()
cp.async(A[1], A_shared[1])       // stage 1 load A
cp.async(B[1], B_shared[1])       // stage 1 load B
cp.async.commit_group()

// Main Loop (steady state):
for k in range(2, K//block_K):
  cp.async.wait_group<0>()        // 等待 stage (k-2) 完成
  __syncthreads()
  gemm(A_shared[(k-2)%2], B_shared[(k-2)%2], C_local)  // compute stage (k-2)
  cp.async(A[k], A_shared[k%2])   // async load stage k (overlapped)
  cp.async(B[k], B_shared[k%2])
  cp.async.commit_group()

// Epilogue: 完成最后 2 个 stage 的计算
cp.async.wait_group<0>(); __syncthreads()
gemm(A_shared[(K-2)%2], B_shared[(K-2)%2], C_local)
cp.async.wait_group<0>(); __syncthreads()
gemm(A_shared[(K-1)%2], B_shared[(K-1)%2], C_local)


// === Hopper (H100) Pipeline (TMA + Warp Specialization) ===
// T.Pipelined 自动推导 warp specialization:

// Producer Threads (by threadIdx.x):
for k in range(K // block_K):
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier(
    A_shared[k%2], &desc_A, [by*block_M, k*block_K], mbar_prod)
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier(
    B_shared[k%2], &desc_B, [k*block_K, bx*block_N], mbar_prod)
  mbarrier.arrive(mbar_prod)     // signal data ready

// Consumer Threads (by threadIdx.x):
for k in range(K // block_K):
  mbarrier.try_wait(mbar_prod)   // wait for producer
  wgmma.fence()
  wgmma.commit_group()
  wgmma.mma_async(A_shared[k%2], B_shared[k%2], C_local)
  wgmma.wait_group<0>()
  // ... producer 已在加载下一 tile (overlapped)


// === AMD CDNA Pipeline ===
s_waitcnt lgkmcnt(0)             // wait for LDS writes
buffer_load_dword lds, ...       // async global→LDS load
s_waitcnt lgkmcnt(0)
// compute ...
```

关键：T.Pipelined 比 Triton 的 num_stages 提供更灵活的 pipeline 控制 — 用户可通过显式标注 producer/consumer 顺序实现自定义 pipeline pattern（如 FlashAttention-3 级别的复杂 warp specialization pipeline）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 TileLang 中，用户只需在循环上添加 `T.Pipelined(K // block_K, num_stages=N)` annotation。编译器自动：(1) 分析 loop body 中各 buffer 的读/写关系 → 确定 prod/cons 角色；(2) 插入对应架构的异步指令序列；(3) Live Variable Analysis → 确定同步点 → 插入 barrier；(4) Hopper 架构上自动应用 warp specialization。对于专家用户，可通过显式 pipeline API（论文未详述）自定义同步策略。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---

## Thread Binding (线程绑定)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Thread Binding 是 GPU kernel 编程中将 tile 级操作和数据映射到具体硬件线程（thread）的过程。在 TileLang 中，Thread Binding 是四个 schedule space 之一，核心挑战是：(1) 确定 block 级 register files 如何划分到各 thread；(2) 推断各 buffer 的 Fragment Layout；(3) 确定循环如何正确 parallelize 以匹配 layout 约束。TileLang 通过 Layout Inference Pass 自动处理 Thread Binding：按优先级层次（GEMM > Element-wise > Copy）逐步推断所有 buffer 的 thread mapping，并在无法自动推断时允许用户通过 T.Fragment 手动指定。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Thread Binding 的推理过程（以 GEMM + bias add 为例，图 7）：
```
// 场景: C_local[4x4] = GEMM(A_shared, B_shared) + D_bias[4]

// Step 1: GEMM (最高优先级) 确定 thread binding
// C_local 的 Fragment Layout:
//   8 threads, 每 thread 持有 2 elements
//   thread 0: {C[0,0], C[0,2]}
//   thread 1: {C[0,1], C[0,3]}
//   thread 2: {C[1,0], C[1,2]}
//   ...
//   每 row 由 2 threads 处理

// Step 2: Infer D_bias layout from C_local's layout
//   由于每 row 的 2 threads 都需要相同 D[row] 元素:
//   D_bias 需要 replicate: 每个 D[row] 复制到 row 对应的 2 threads
//   Fragment Layout for D: f(row_idx) → (thread_id, reg_id)
//     thread 0,1: D[0] in reg 0 (replicated)
//     thread 2,3: D[1] in reg 0 (replicated)
//     ...

// 生成伪代码:
for tx in T.Parallel(8):    // 8 threads
  for i in T.vectorized(2):  // per-thread vectorized
    C_local[tx//2, (tx%2)*2 + i] += D_bias[tx//2]  // D 已 replicate to each thread
```

图 8 展示了更复杂场景的 multi-stage Thread Binding Inference：T.copy 操作先扩展为多个 loop axes → Layout Inference Pass 自动 parallelize + vectorize → Layout Swizzling 应用 → 最终生成高效 memory access pattern。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 TileLang 中用户通常无需手动处理 Thread Binding — T.Kernel(threads=N) 确定总线程数，Layout Inference 自动处理映射。专家可通过 T.Fragment 和 InferLayout 接口手动定义 thread→buffer 映射策略。T.Parallel 原语可显式标注并行循环维度。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---

## Swizzle Layout (内存布局置换)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Swizzle Layout（内存布局置换）是一种 GPU shared memory 地址重映射技术，通过 XOR（异或）位操作将逻辑上连续的 shared memory 地址映射到不同物理 bank，以避免 bank conflict。在 NVIDIA GPU 上，shared memory 被组织为 32 个 bank（每个 bank 4 bytes），同一 warp 内 32 个 thread 如果在同一 cycle 访问同一 bank 的不同地址，会触发 bank conflict（最多 32-way），导致访问被串行化，带宽下降。Swizzle Layout 通过对地址高位和低位做 XOR 操作，打散连续访问的 bank 分布，确保相邻线程访问不同 bank。TileLang 通过 Layout Composition 机制将 SwizzleLayout 作为 built-in strategy，T.gemm 默认对 A_shared 和 B_shared 应用 SwizzleLayout。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Swizzle Layout 在 GEMM shared memory tile 中的效果：
```
// 无 Swizzle（4-way bank conflict 示例）:
// 32 threads × 128-bit (4 floats) load → row-major 连续地址
// Thread 0: addr 0   (bank 0)
// Thread 1: addr 4   (bank 1)
// ...
// Thread 8: addr 32  (bank 0) ← conflict with thread 0! (周期 8 × 4 bytes = 32 = bank repeat)

// 有 Swizzle（T.gemm 默认）:
// XOR(bits=3) swizzle: addr' = addr XOR (addr >> 3) & 0x7
// Thread 0: addr 0  → bank 0
// Thread 1: addr 4  → bank 1
// ...
// Thread 8: addr 32 → swizzle → addr' = 32 XOR (32>>3)&7 = 32 XOR 4 = 36 → bank 9  ← no conflict!
// 结果: 0 bank conflict → shared memory bandwidth = peak
```

TileLang 的 Swizzle 实现通过 Layout Composition：
```
base_layout = Layout(shape=[M, K], strides=[K, 1])  // row-major
swizzle_layout = SwizzleLayout(bits=3, dim=1)        // 沿 K 维 swizzle
final_layout = base_layout ∘ swizzle_layout           // composition
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Swizzle Layout 在 CUTLASS 中通过 `cutlass::layout::ColumnMajorSwizzle` 或 `RowMajorSwizzle` 实现。TileLang 中 T.gemm 自动应用，用户可通过 T.annotate_layout 覆盖自定义 swizzle 模式。T.use_swizzle(10) 是一种不同的 swizzle — 它作用于 thread block scheduling ordering（而非 shared memory 地址），通过打乱 thread block 的执行顺序优化 L2 cache locality（相邻 thread block 可能访问重叠的 global memory 数据区域）。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---

## Intrinsic Tensorization (硬件指令张量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Intrinsic Tensorization 是 TileLang 中利用 GPU 硬件特殊指令（Tensor Cores mma、DP4A、IMAD 等）进行高性能矩阵运算的机制。现代 GPU 提供多种精度和吞吐量的指令路径：例如 NVIDIA RTX 3090 上，IMAD（标量融合乘加）≈ 17.8 TOPS，DP4A（向量化 4 元素点积）≈ 71.2 TOPS，MMA（Tensor Core 矩阵乘法）≈ 284 TOPS。选择合适的指令取决于输入 shape 和 data type。TileLang 提供两种互补的 Tensorization 方式：(1) Tile Library-based — 通过 T.call_extern 调用 CUTLASS cute (NVIDIA) 或 Composable Kernel (AMD) 的 tile API（如 cute::gemm_ss），自动选择最优指令；(2) Direct PTX/C++ injection — 通过 T.ptx 直接发射内联 PTX 指令（如 mma.m16n8k32.row.col.s32.s8.s32），或通过 T.import_source + T.call_extern 注入 C++ 模板实现的 DP4A/IMAD 等指令。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

TileLang 中两种 Tensorization 方式的对比：
```
// 方式 1: Tile Library-based (默认，图 10c)
T.gemm(A_shared, B_shared, C_local)
// ↓ Lower to:
T.call_extern("cute::gemm_ss", 
  A_shared[SwizzleLayout], B_shared[SwizzleLayout], C_local[MMA_MatrixLayout])
// ↓ Codegen to CUDA C:
cutlass::gemm_ss<128, 128, 32, 2, 2>(AShared, BShared, C_local);
// CUTLASS 内部自动选择 mma.m16n8k32 或 wgmma.m64n64k16 等最优指令

// 方式 2: Direct PTX injection (图 10b，专家使用)
T.ptx("mma.m16n8k32.row.col.s32.s8.s32", 
      C_local_regs, A_regs, B_regs, C_local_regs)
// ↓ 直接生成 PTX:
asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s32.s32 "
             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
             : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
             : "r"(a0),"r"(a1), "r"(b0), "r"(c0),"r"(c1),"r"(c2),"r"(c3));

// 方式 3: C++ Source Injection (图 10a，低精度场景)
T.import_source("dp4a_kernel.cuh")  // 注入 C++ 模板实现
T.call_extern("dp4a_kernel<int8_t>", A, B, C)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

默认使用 Tile Library-based 方式（简单、vendor-optimized 性能），但其局限是：(1) cute::gemm_ss 内部管理 shared→register data flow，外部无法 annotate/override layout；(2) CUTLASS 模板展开占约 90% TileLang 编译时间（NVCC 12.8 trace 验证）。Direct PTX/Injection 方式提供完全控制但需用户为每种硬件指令实现完整的指令集封装。TileLang 计划未来构建 self-hosting Tile Library 用 TileLang 自身替代 CUTLASS 依赖，同时保留 Direct Injection 路径。对于 AMD GPU，TileLang 使用 Composable Kernel (CK) 和手写 HIP 代码。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

