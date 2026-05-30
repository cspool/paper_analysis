## Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是Stream-K——一种work-centric的GEMM并行分解策略，将MAC-loop迭代（而非输出tile）作为跨SM的workload量子化单元。核心实现包括：(1) 将GEMM的总MAC-loop迭代次数（total_iters = ⌈m/BLK_M⌉ × ⌈n/BLK_N⌉ × ⌈k/BLK_K⌉）均匀分配给g个CTA，每个CTA执行iters_per_cta = ⌈total_iters/g⌉个MAC-loop迭代；(2) CTA的迭代范围沿m→n→k线性化连续映射，可跨越output tile边界；(3) 当一个CTA的起始/结束迭代不与tile边界对齐时，通过temporary global storage交换partial sums，由执行该tile的k=0 MAC-loop迭代的CTA负责累积并写出最终结果；(4) "two-tile Stream-K + data-parallel"混合调度——限制iteration balancing仅在最后部分数据并行wave的tile范围内，确保每个输出tile最多被两个CTA覆盖，隐藏inter-CTA同步延迟并改善cache locality；(5) 基于解析模型的grid size选择——建模CTA运行时间为固定开销a + partial sum输出开销b + 每迭代MAC开销c + 每协作CTA的partial sum累积开销d，最小化该模型以选择最优g。

  实验比较：(1) vs 同blocking factor的data-parallel CUTLASS kernel（衡量量化效率提升）；(2) vs cuBLAS ensemble (CUDA 11.6)；(3) vs CUTLASS oracle（始终选择最优data-parallel blocking factor的理想化ensemble）。FP64 oracle从5种blocking中选择，FP16→32 oracle从4种blocking中选择。评估覆盖32,824个GEMM shapes（m,n,k ∈ {128...8192}，对数采样，计算体积跨越六个数量级）。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（108 SM cores），功率锁定400W，SM时钟锁定1005 MHz（~71%动态峰值）。FP64 Tensor Core峰值吞吐13.9 TFLOP/s，FP16→32 Tensor Core峰值吞吐222.3 TFLOP/s。Software: CUDA 11.8, CUTLASS 2.11。

- 评估性能的软件/脚本是什么。修改了什么。
  CUTLASS library（https://github.com/NVIDIA/cutlass）。修改：(1) 在CUTLASS的CTA-wide MacLoop() subroutine之上实现Stream-K grid-level decomposition（Algorithm 5），将total_iters均匀分配到g个CTA，每个CTA沿m→n→k线性化执行其迭代范围；(2) 实现partial sum的temporary global storage交换和accumulation逻辑——使用StorePartials/LoadPartials/Signal/Wait原语进行inter-CTA同步；(3) 实现"two-tile Stream-K + data-parallel"混合调度策略；(4) 实现基于解析模型的grid size选择启发式——参数a/b/c/d通过微基准（microbenchmark）每target architecture一次性经验测量后静态编译入库。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/NVIDIA/cutlass（自CUTLASS 2.11起包含Stream-K实现）。使用CUDA 11.8编译可复现论文性能。

  评估原理：
  1. 为每种精度（FP64, FP16→32）构建单个Stream-K kernel（使用"two-tile Stream-K + data-parallel"混合分解）。
  2. 对32,824个GEMM shapes逐一执行warmup + timing测量，计算GFLOPS/s和处理器利用率百分比。
  3. CUTLASS data-parallel baseline使用相同blocking factor的data-parallel kernel；cuBLAS使用CUDA 11.6的ensemble；CUTLASS oracle对每个shape枚举所有候选blocking factor取最优。
  4. 绘制roofline图（利用率% vs 计算强度ops/byte）展示性能响应的一致性（narrow band = better）。

  全过程（以FP16→32 GEMM, m×n×k = 384×384×128, BLK_M=128, BLK_N=128, BLK_K=32, g=4 CTAs为例）：
  ```
  Host: 调用Stream-K GEMM kernel
  
  Step 1: Grid size selection (启发式模型)
    total_tiles = ⌈384/128⌉ × ⌈384/128⌉ = 3 × 3 = 9
    iters_per_tile = ⌈128/32⌉ = 4
    total_iters = 9 × 4 = 36
    → 选择 g = 4 CTAs（最优由解析模型确定）
    → iters_per_cta = ⌈36/4⌉ = 9

  Step 2: Grid launch (4 CTAs on 4 SMs)
    CTA_0: iter ∈ [0, 9)   → tile 0 (iter 0-3) + tile 1 (iter 4-7) + tile 2 start (iter 8)
    CTA_1: iter ∈ [9, 18)  → tile 2 end (iter 12-13) + tile 3 (iter 12-15) + tile 4 (iter 16-17) + ...
    CTA_2: iter ∈ [18, 27) → ...
    CTA_3: iter ∈ [27, 36) → ...

  Step 3: Per-CTA execution (以CTA_0为例)
    while iter < 9:
      tile_idx = iter / 4 = 0
      local_iter = 0, local_iter_end = min(9, 4) - 0 = 4
      
      MacLoop(tile_0, iter=0, iter_end=4): 完整执行tile 0的4个MAC-loop迭代
        for iter in 0..4:
          kk = iter × 32
          frag_a = LoadFragment(A, mm=0, kk)  // 128×32
          frag_b = LoadFragment(B, kk, nn=0)  // 32×128
          MMA: accum += frag_a × frag_b        // 128×128×32 MACs per iter
      
      tile_started=true → 不写partials（k=0被CTA_0覆盖）
      tile_ended=true → 直接StoreTile(C, tile_0, accum)
      
      iter = 4
      → tile_idx = 1, local_iter = 0, local_iter_end = min(9, 8) - 4 = 4
      MacLoop(tile_1, 0, 4): 完整执行tile 1
      → StoreTile(C, tile_1, accum)
      
      iter = 8
      → tile_idx = 2, local_iter = 0, local_iter_end = min(9, 12) - 8 = 1
      MacLoop(tile_2, 0, 1): 仅执行tile 2的第1个MAC-loop迭代
      
      tile_started=true → 不写partials（CTA_0覆盖了k=0）
      tile_ended=false → 等待CTA_1完成tile 2的剩余iterations
        Wait(flags[1])
        accum += LoadPartials(partials[1])
      → StoreTile(C, tile_2, accum)

  Step 4: Inter-CTA partial sum consolidation
    CTA_0执行tile_2的k=0迭代时仅做了1个MAC-loop迭代
    CTA_1执行了tile_2的后3个MAC-loop迭代（iter 9-11, local=1-3）
      → CTA_1: StorePartials(partials[1], accum) + Signal(flags[1])
      → CTA_0: Wait(flags[1]) + accum += LoadPartials(partials[1])

  Step 5: 性能输出
    → 100% quantization efficiency（所有4个SM执行相同9个MAC iterations）
    → vs data-parallel (9 CTAs for 9 tiles on 4 SMs = 3 waves, last wave 1/4 active = 75% utilization):
      4/3× 理论利用率提升
    → 实测FP16→32平均1.63× vs CUTLASS data-parallel, 最大14.7×（极端强伸缩scenario m×n小k大）
    → vs cuBLAS FP16→32: 平均1.13×, 最大6.74×
    → vs CUTLASS oracle: 平均1.12× (FP16→32), 证明Stream-K达到tile-based方法无法企及的利用率
  ```

  关键设计要点：
  - Stream-K的单位workload（1个MAC-loop iteration = BLK_M×BLK_N×BLK_K MACs）比data-parallel的单位（1个output tile = ⌈k/BLK_K⌉ MAC-loop iterations）小32×，因此量化效率远优于tile-based方法
  - Communication/synchronization/global storage overheads与问题规模无关，仅与CTA数g（处理器宽度）成正比，即O(p)
  - 当tile数 > CTA数时，每个tile最多被2个CTA覆盖，synchronization-waiting因producer-consumer时间偏移自然隐藏
  - 单kernel、单tile size配置即可超越需要复杂heuristics的20+ kernel ensemble（cuBLAS），可执行代码体积减少20×
