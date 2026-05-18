## Segment Sum Shared-Memory Reduction for GPU SpMM

术语是什么？

Segment Sum Reduction 是 Swift 在 regular SpMM kernel 中使用的一种 shared-memory 局部规约技术，用于在 global atomicAdd 之前先将同 row_idx 的 partial product 在 warp/shared memory 内部合并，从而大幅减少 atomic 操作次数和地址冲突。Swift 维护 positionIdx（记录每个非零元在其列内的位置，用于判断是否为 segment 起点）和 offsetIdx（记录 segment 在 shared memory 中的偏移量）两套辅助索引，指导 warp 内线程对 rowIdx 相同的 partial sum 做 segment-level 局部求和。

从kernel调度角度拆解术语：

```
// Swift Regular Kernel 的 segment sum 流程
// 输入: smem[0..31] 含 32 个 (row_idx, partial) pairs
//       每个线程已将其 partial product 写入 smem[lane_id]

// positionIdx[i] = 该非零元在所在列内的位置偏移
//   若 positionIdx[i] == 0 → 该非零元是所在列的第一个非零元 → segment 起点
// offsetIdx[i] = 该非零元对应的 segment 在归约 buffer 中的偏移量

// Step 1: 识别 segment 边界
is_segment_start = (positionIdx[lane_id] == 0)

// Step 2: 将 segment start 标记写入 shared memory
__syncthreads()
smem_flag[lane_id] = is_segment_start

// Step 3: Warp-level segment sum
// 按 row_idx 分组: 将同 row_idx 的 partial 累加到 offsetIdx 指定的 buffer 位置
for step = 0 to warpSize-1:
    if lane_id == step:
        seg_offset = offsetIdx[step]
        smem_buf[seg_offset] += smem_partial[step]

__syncthreads()

// Step 4: 仅 segment 起点线程写回
if is_segment_start:
    seg_offset = offsetIdx[lane_id]
    row = rowIdx[lane_id]
    atomicAdd(&C[row * N + j], smem_buf[seg_offset])
```

对比 naive 方法（每个线程都做 atomicAdd）：
- Naive: 32 个 atomicAdd 调用 → 大量 address contention → 数据加载开销 >32%
- Segment sum: 仅 segment 起点线程做 atomicAdd（通常 << 32）→ 大幅减少 atomic 冲突
- 例如：32 个非零元中 16 个指向同 1 个 row → naive 需 32 atomics（16 个竞争同一地址），segment sum 仅 1 次 atomic（16 个 partial 已在 SMEM 合并）

术语一般如何实现？如何使用？

Segment sum 的实现要素：
1. **辅助索引**：positionIdx 和 offsetIdx 在 CPU 预处理阶段构建，基于排序后 A 的列结构计算。对于 regular block（32 列），每列内非零元的 positionIdx 从 0 递增；offsetIdx 由同 row_idx 的首次出现位置决定。
2. **Shared memory 使用**：regular kernel 的 thread block 需要额外 SMEM 用于 segment sum buffer（通常 32×sizeof(float) = 128B per warp + flag 数组）。
3. **同步点**：segment start 标记写入后需要 `__syncthreads()`；segment sum 完成后需要 `__syncthreads()` 再写回。这些同步开销需要被 coalesced 加载带来的收益抵消。
4. **适用性**：segment sum 在 regular part 最有效，因为 regular block 的列结构规整，positionIdx/offsetIdx 紧凑。irregular part 因为列长分布不规则，仍用 stride-based 遍历 + 直接 atomicAdd。
5. **与前序工作的关系**：此策略与 hash-based write-back (VDHA) 的 shared-memory hash aggregation 共享"先局部聚合再写回"的思想，但实现不同：Segment sum 用确定性索引（positionIdx/offsetIdx）而非 hash probing，无需处理 hash collision 和 fallback。

涉及论文标题：
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

