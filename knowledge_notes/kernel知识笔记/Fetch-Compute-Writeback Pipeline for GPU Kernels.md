## Fetch-Compute-Writeback Pipeline for GPU Kernels

术语是什么？

Fetch-compute-writeback pipeline (取数-计算-写回流水线) 是一种 GPU kernel 执行重叠技术：将传统 fetch→writeback 两阶段重构为 fetch→compute (hash aggregation)→writeback 三阶段，通过 double buffering 和异步 copy (cp.async) 实现当前 segment 的 hash computation 与下一 segment 的 memory fetch 重叠，隐藏 hash computation latency。该技术利用 write-back 阶段的高 memory stall（>45% long scoreboard waits）——即使高 occupancy 也无法隐藏 uncoalesced memory access latency，因此可在 memory stall 期间执行有用计算。

从kernel调度角度拆解术语：

VDHA pipeline 伪代码：

```
Input: segments seg[]; SMEM buffers buf[0], buf[1]; hash table H

cp.async.Fetch(seg[0], buf[0])               // 异步加载首个 segment
__syncthreads()
for i = 0 to N_segs-1:
    if i != N_segs-1:
        cp.async.Fetch(seg[i+1], buf[(i+1)%2]) // 异步预取下一 segment
    indices, values ← buf[i%2]                  // 当前 segment 已在 SMEM
    for each (row, mat_val) in segment:
        hash_insert(H, row, mat_val × vec_val)  // hash aggregation
    __syncthreads()
    if hash_full(H) or i == N_segs-1:
        flush(H, y)                    // bucket-order flush
    cp.async.wait_group()               // 确保下一 segment 已就绪
```

Timeline 重叠示意：
```
Seg 0: [Fetch seg0][===== Hash Compute seg0 =====][Flush]
Seg 1:             [Fetch seg1][===== Hash Compute seg1 =====][Flush]
Seg 2:                         [Fetch seg2][===== Hash Comp... ]
                   ↑ overlap: fetch seg(i+1) || hash compute seg(i)
```

效果：stall ratio >45%→~15%，hash computation cost 16.7%→12.3%。

术语一般如何实现？如何使用？

需要 GPU 架构支持异步 copy（NVIDIA Ampere SM80+ cp.async，AMD asynchronous copy units）。Double buffering 需要足够 SMEM 容纳两个 buffer + hash table。效率取决于 compute 和 memory fetch 的耗时比例——memory stall 越严重 overlap 收益越大。类似技术广泛用于 FlashAttention async copy、SpGEMM kernel memory-compute overlap、general GPU kernel optimization。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

