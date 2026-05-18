## Hash-based Aggregation for SpMSpV Write-back

术语是什么？

Hash-based aggregation (基于哈希的聚合) 是 GPU SpMSpV 中 write-back 阶段的一种策略：用每个 CTA 私有的 shared-memory hash table 暂存和累积 partial products，在 hash table 接近满时批量 flush 到 global memory。相比 atomic write-back（每个 partial product 发 global atomicAdd）和 sort-based write-back（全局排序后 reduce），hash-based 方法通过 local aggregation 减少 global write conflicts，并通过 bucket-order flush 改善 memory coalescing。该策略受 SpGEMM 中 hash-based accumulation 启发——SpGEMM 每个 output row 维护小 hash table 消除 intra-row conflicts。但 SpMSpV 缺少 SpGEMM 的自然 row partitioning，所有 intermediate updates 汇聚到单一 output vector，因此 hash table 只能消除 intra-block conflicts，cross-block conflicts 仍需 global atomics。

从kernel调度角度拆解术语：

VDHA 的 hash-based insertion 伪代码：

```
function Insert(H, idx, val):
    h ← idx % TABLE_SIZE              // modulo hash，保留 row index 低位
    cnt ← 0
    while cnt < FALLBACK_ITER:
        old ← atomicCAS(&H.key[h], -1, idx)   // -1 = empty slot
        if old == -1 or old == idx:
            UpdateHash(H.val[h], val)          // accumulate value
            return
        h ← (h + STRIDE) % TABLE_SIZE          // linear probing with fixed stride
        cnt ← cnt + 1
    Fallback(idx, val)               // probe 超限 → global atomicAdd fallback
```

关键设计：
- **Modulo hash**: idx % TABLE_SIZE，保留低位使 bucket order flush 改善 coalescing
- **Linear probing with stride**: (h + C) % TABLE_SIZE，相比 (h+1) 降低 locally distributed nonzeros 碰撞概率
- **FALLBACK_ITER**: 限制 probing 次数，防止无限循环导致 warp divergence
- **Update 策略**: 同列 segment 直接用 `H.val[idx] += val`（无 atomic）；跨列需 atomicAdd
- **局部聚合率 ρ(T) = 1 - F(T)/N**：衡量 hash table 吸收的 update 比例。在 it-2004 上 column decomposition+reorder 后 ρ 从 51.0% 提升至 89.8% (T=2048, density=100%)
- **Table size**: 2048 entries (16KB)，平衡 aggregation 效果和 occupancy

术语一般如何实现？如何使用？

每个 CTA 维护独立 shared-memory hash table（不跨 CTA 共享）。`atomicCAS` 实现 lock-free concurrent insertion。Linear probing 提供良好 GPU cache locality。Fallback 保证正确性同时避免无限循环。有效性依赖矩阵结构：需 sufficient temporal locality (ρ) 使 aggregation 收益超过 hash overhead。VDHA 证明通过 column decomposition + reordering + fetch-compute-writeback pipeline，hash-based write-back 在 web graphs 上实现 1.41× geomean speedup。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

