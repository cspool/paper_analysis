- baseline方法是什么？
  Baseline是现有的GPU SpMSpV write-back策略：(1) **Atomic write-back**：每个partial product直接用global atomicAdd写入output vector y[ind]。this method在many-to-one scatter pattern下产生严重address contention，uncoalesced memory stores导致带宽利用率极低。在A100上sparsity=0.1时atomic write-back仅270 GB/s (peak 1555 GB/s的17%)，write-back占overall runtime的>30%，stall cycles中>45%是long scoreboard waits。(2) **Sort-based write-back**：buffer所有(row_idx, val) pairs→global sort by row index→sequential reduce duplicates→每个row仅写一次。sort阶段带宽仅~43.3 GB/s，write-back占overall runtime的>70%，需要large temporary buffers。
  
  全栈执行例子（以it-2004 web graph, sparsity=0.1, NVIDIA A100为例）：
  - 算法层：SpMSpV y=A*x，CSC格式矩阵→vector-driven (column-major) paradigm: 对x中每个nonzero遍历A对应column→生成partial products→write-back到y
  - 系统框架层：GPU graph frameworks (Gunrock/GraphBLAST)提供atomic-based SpMSpV kernel，或Adaptive SpMSpV根据matrix statistics在4种column-major kernel+2种row-major kernel间选择
  - 编译框架层：论文未明确说明（手工CUDA kernel，无编译框架自动生成）
  - kernel调度层：BlockAtomic (Gunrock-like): 多short column聚合到一个CTA→CTA内threads各自计算partial products→global atomicAdd写入y。GlobalAtomic: global prefix scan计算total NNZ→均匀分配每个CTA→CTA内threads global atomicAdd。BlockSort (FastSpMSpV-like): CTA内生成(row_idx, val) pairs→sort→reduce→coalesced global write。GlobalSort: global均匀分配→sort-reduce→write。Load balancing策略：Block-mapped按column数分配CTA（skewed分布下poor balance），Global-mapped按NNZ均匀分配CTA（较好balance但prefix-scan overhead）
  - 硬件架构层：NVIDIA A100 GPU。global memory bandwidth 1555 GB/s，L2 cache 40MB，per-SM 168KB shared memory。Atomic units处理global atomicAdd，uncoalesced scatter导致cache line浪费、L2 hit rate低、bandwidth saturation差

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**VDHA (Vector-Driven Hash Aggregation)**：通过shared-memory hash table做local aggregation减少global write conflicts，column decomposition+reordering增强locality，fetch-compute-writeback pipeline隐藏hash overhead。

  论文方法全栈执行例子（以it-2004, sparsity=0.1, A100为例）：
  - 算法层：SpMSpV y=A*x，CSC格式→vector-driven paradigm，hash-based write-back替代atomic/sort write-back
  - 系统框架层：VDHA CUDA kernel可集成到adaptive SpMSpV框架中（配合预测模型在VDHA和baseline间选择，best-of-7 fallback实现1.22× speedup）
  - 编译框架层：论文未明确说明（手工CUDA kernel，使用cp.async异步copy指令、atomicCAS等PTX级操作，无编译框架自动生成）
  - kernel调度层（核心创新）：
    (1) **Shared-memory hash aggregation**（解决baseline缺陷：global atomic scatter→low bandwidth utilization）：
    每个CTA维护2048-entry shared-memory hash table→partial products插入hash table做local accumulation→hash table满时bucket-order flush到global memory。对比baseline：BlockAtomic每个update都global atomic→>30% runtime in write-back with ~270 GB/s bandwidth；VDHA大部分updates在shared memory local aggregation→仅flush和fallback才global write→global atomic conflicts显著减少（atomic-unit utilization从22.99%降至12.82%）。hash table还提供partial ordering：entries按bucket order (0→2047) flush→warp内threads访问相对连续地址→改善memory coalescing（γ从0.744提升至2.607）。
    (2) **Column decomposition with reordering**（解决baseline缺陷：skewed column lengths→workload imbalance + poor hash locality）：
    长列按SPLIT_SIZE=256切分为segments→segments metadata按首row index排序（仅排序segment metadata而非nonzeros本身，O(S log S) cost where S<<N）→cross-column segment overlap增强hash table reuse。对比baseline：Global/Block mapping均不exploit long-column内部缺乏locality的特点→ρ仅51.0% (T=2048, density=100%)→hash table insufficient aggregation；VDHA切分+重排序后ρ提升至89.8%→更多updates在共享内存完成aggregation。local overlap ratio ρ从0.510→0.898，coalescing factor γ从0.744→2.607。
    (3) **Fetch-compute-writeback pipeline**（解决baseline缺陷：high memory latency→warp stall无法被occupancy掩盖）：
    double buffering + cp.async异步fetch下个segment→当前segment做hash aggregation→next segment ready时swap buffer。对比baseline：45%+ stall cycles是long scoreboard waits (pending global memory)→即使高occupancy也无法隐藏latency；VDHA将hash computation叠加到memory fetch latency上→stall ratio从>45%降至~15%→hash computation cost从16.7%降至12.3%。
  - 硬件架构层：NVIDIA A100 GPU。利用shared memory（per-SM 168KB）做hash table显式scratchpad（vs L1 cache implicit caching），2048-entry hash table=16KB per CTA→8 CTAs/SM→256 threads/CTA平衡occupancy。cp.async (Ampere+)实现asynchronous global→shared copy。atomicCAS支持intra-CTA shared memory atomic操作。FALLBACK_ITER机制控制linear probing worst-case latency避免warp divergence。

  Baseline缺陷→VDHA方案映射：
  | Baseline缺陷 | VDHA方案 | 效果 |
  |-------------|---------|------|
  | Global atomic scatter→low bandwidth (~270 GB/s) | Shared-memory hash local aggregation + coalesced flush | Atomic-unit utilization ↓ 22.99%→12.82% |
  | Skewed column lengths→imbalance + poor hash locality | Column split (SPLIT_SIZE=256) + segment reorder by first row index | Local overlap ρ ↑ 0.510→0.898, γ ↑ 0.744→2.607 |
  | Memory stall dominates (>45% long scoreboard) | cp.async double-buffering pipeline: fetch↔compute overlap | Stall ratio ↓ >45%→~15%, hash cost ↓ 16.7%→12.3% |
  | Sort-based→O(N log N) sort overhead (~43 GB/s sort bandwidth) | Hash aggregation O(N) with hash table (O(1) amortized insertion) | No global sort needed, hash computation mostly hidden |
  | No basis for kernel selection→suboptimal choice | Decision tree predictor (5 features, 91.3% accuracy) → fallback to best-of-7 | Adaptive geomean speedup 1.13×→1.22× on SuiteSparse |