## Hierarchical Memory Spilling in Kernel Fusion（内核融合中的层次化内存溢出）

术语是什么？通过联网搜索让回答具体和精准。

Hierarchical Memory Spilling 是 FlashFuser Dataflow Analyzer 中的核心技术，用于在 fused GEMM kernel 中将 reused tensor（中间结果）从高层缓存逐步溢出到低层缓存。与传统的单 SM 内 fusion（中间结果必须完全放入 register 或 SMEM，否则 fusion 失败）不同，FlashFuser 的 spilling 策略允许中间 tensor 跨越 reg→SMEM→DSM→global memory 四层。贪心算法从最高层缓存（寄存器）开始分配，容量不足时剩余部分逐层下溢。这解决了单 SM SMEM 约 227KB 容量限制导致的 fusion failure 问题——当 C[M×N] tile 超出 SMEM 时，不再直接放弃 fusion，而是溢出一部分到 DSM（cluster 内多 SM SMEM 聚合，最大约 3.6MB）。

从编译框架角度拆解术语：

```
Dataflow Analyzer 的贪心 spilling 算法 (Algorithm 1):
Input: tensor footprint DF, memory hierarchy [reg, SMEM, DSM, global]
Output: per-memory-level allocation mapping, data movement volume DV

remaining ← DF
foreach level in [reg, SMEM, DSM, global]:
  if remaining ≤ 0: break
  alloc ← min(remaining, level.capacity)
  mapping[level] ← alloc
  remaining ← remaining - alloc
  // 计算该层 data movement
  DV[level] ← update_dv(tile_size, DF)  // 考虑 tiling 带来的额外访问

例: C tile = 512KB
  reg: alloc = min(512KB, ~64KB per thread block) = 64KB, remaining = 448KB
  SMEM: alloc = min(448KB, 227KB) = 227KB, remaining = 221KB
  DSM: alloc = min(221KB, ~3.6MB cluster) = 221KB, remaining = 0
  global: alloc = 0 (不再溢出)

结果: C tile 分布在 reg(64KB) + SMEM(227KB) + DSM(221KB)，不需 global memory round-trip
```

溢出的代价：每层 memory 有不同带宽（reg > SMEM > DSM > global），溢出到更低层增加数据传输时间。Dataflow Analyzer 计算各层 data movement volume，供 cost model 做 minimax 优化，避免任何一层成为瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashFuser 的 spilling 机制实现方式：
1. **Footprint 计算**：`GetFootprint(t.block)` 确定单 block tile 的数据访问量
2. **I/O tensor 处理**：对输入/输出 tensor，直接计算从 global memory 搬运的总量（遍历所有相关维度，乘以 tiling factor）
3. **Reused tensor 处理**：贪心分配到 hierarchy [reg → SMEM → DSM → global]，每层 capacity 从硬件 spec 获取
4. **Capacity 约束**：Rule 5 (Memory Capacity Limit) 要求 tensor 不能超过最低层缓存（global memory）的容量——这是逻辑上可以 spilling 到的最低层
5. **代码生成对应**：spilling plan 在 backend codegen 中体现——register 部分用变量直接持有，SMEM 部分用 `__shared__` 声明，DSM 部分通过 cluster-wide shared memory + mbarrier 同步访问

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
