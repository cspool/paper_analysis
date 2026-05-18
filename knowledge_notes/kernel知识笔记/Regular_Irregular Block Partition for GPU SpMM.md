## Regular/Irregular Block Partition for GPU SpMM

术语是什么？

Regular/Irregular Block Partition 是 Swift 提出的 GPU SpMM 预处理策略。在按 NNZ 排序稀疏矩阵 A 的列之后，Swift 将 A 按 warpSize=32 划分为两类 block：(1) Regular block：列宽恰好等于 32 的连续列组，每个 block 由 1 个 warp (32 threads) 处理；(2) Irregular block：不足 32 列的短列组或超过 warpSize 的长列残留元素。两类 block 用不同的 kernel 路径处理——regular kernel 优先追求 memory coalescing（通过 segment sum 减少 atomic），irregular kernel 优先追求 load balancing（通过 sub-column 拆分避免单 warp 阻塞）。

从kernel调度角度拆解术语：

```
// Swift 预处理: blocking 阶段
// 输入: A 已按列 NNZ 升序排序的 CSC 格式
// 输出: regular 和 irregular 两套索引结构

block_start = 0
regular_count = 0
irregular_count = 0

while block_start < K:  // K = A 的列数
    // 尝试构成 width=32 的 regular block
    if block_start + 32 <= K:
        // Regular block: 32 个连续列
        blkPtr[regular_count] = nnz_offset            // block 在 value/rowIdx 中的起点
        blkColIdx[regular_count] = block_start        // block 起始列号
        // 为 block 内非零元生成 positionIdx, offsetIdx (用于 segment sum)
        regular_count++
        block_start += 32
    else:
        // Irregular part: 剩余不足 32 列
        break

// 剩余列 + 未进入 regular 的长列残块 → irregular part
for col = block_start to K-1:
    nnz_in_col = colPtr[col+1] - colPtr[col]
    if nnz_in_col > SPLIT_THRESH:  // 长列拆分
        num_subcols = ceil(nnz_in_col / SPLIT_SIZE)
        for s = 0 to num_subcols-1:
            irrPtr[irregular_count] = colPtr[col] + s * SPLIT_SIZE
            colIdxIndex[irregular_count] = col       // 记录原列号
            blkStart[irregular_count] = s * SPLIT_SIZE
            blkStop[irregular_count] = min((s+1)*SPLIT_SIZE, nnz_in_col)
            irregular_count++
    else:  // 短列直接作为一个 block
        irrPtr[irregular_count] = colPtr[col]
        irregular_count++
```

两类 kernel 的调度差异：
- **Regular kernel**: thread block = 32×8 (8 warps)，每 warp 负责 1 个 regular block (32 列)。执行路径固定：coalesced 加载 → shared memory segment sum → 少量 atomicAdd。
- **Irregular kernel**: warp 动态调度——通过 colIdxIndex 判断当前任务是独立短列还是长列子块 → blkStart/blkStop 定位范围 → lane 以 stride=32 遍历范围 → 对每个非零元循环访问 B 的 N 列 → atomicAdd 写回。

术语一般如何实现？如何使用？

Blocking 的实现要点：
1. **warpSize 耦合**：regular block width=32 与 CUDA warp size (32) 严格对齐，保证每线程一列的 1:1 映射。
2. **索引结构**：Regular 需要 blkPtr、blkColIdx、positionIdx、offsetIdx。Irregular 需要 irrPtr、irrValue、irrRowIdx、colIdxIndex、blkStart、blkStop。
3. **预处理开销**：blocking 需要在 CPU 端完成，包括列分类和索引生成。NNZ > 10^6 时预处理成本可能超过或接近 Sputnik/RoDe 的开销。
4. **消融收益**：irregular part 的 load-balancing 优化（相对 naive 单 warp 处理长列）在 N=32 和 N=128 下分别带来 2.26× 和 2.69× 几何平均 speedup。
5. **适用条件**：非零元分布较均匀时 blocking 效果最好；当非零元集中在少数 32×32 block 中时收益下降，此时 ASpT 的 adaptive tiling 可能更优。

涉及论文标题：
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

