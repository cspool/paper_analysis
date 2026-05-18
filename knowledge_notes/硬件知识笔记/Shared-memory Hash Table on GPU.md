## Shared-memory Hash Table on GPU

术语是什么？

GPU shared-memory hash table (共享内存哈希表) 是一种将 GPU on-chip shared memory 作为显式管理的哈希表 scratchpad 的技术。与依赖 L1/L2 cache 的 implicit caching 不同，programmer 直接在 shared memory 中分配 key-value 数组，通过哈希函数和 probing 策略实现 lock-free concurrent insertion/lookup/accumulation。每个 CTA (Cooperative Thread Array) 维护私有 shared-memory hash table，CTA 内所有线程可并发访问。相比 global memory hash table，shared memory 延迟低两个数量级（~20 cycles vs ~200-800 cycles），带宽高一个数量级（~1.5 TB/s per SM vs ~1.5 TB/s whole GPU）。典型应用包括 SpGEMM 的 intra-row accumulation、SpMSpV 的 write-back aggregation、图处理的 frontier 去重。

从硬件架构角度拆解术语：

NVIDIA A100 上 shared-memory hash table 的硬件资源约束与运转流程：

```
SM (Streaming Multiprocessor) 资源：
- Shared memory per SM: 168 KB (configurable partition from L1/SMEM 192KB total)
- Max warps per SM: 64 (2048 threads)
- Max CTAs per SM: 32

VDHA 的 shared-memory hash table 分配：
- Table size: 2048 entries × (4B key + 4B value) = 16 KB per CTA
- Double buffering: 2 × segment buffer (variable size, ~4-8 KB each)
- Total SMEM per CTA ≈ 16 + 8 + 8 = 32 KB
- CTAs per SM ≈ 168/32 ≈ 5-8 CTAs (平衡 occupancy)
- Threads per CTA = 256 (8 warps), 确保 full warp utilization

Hash table 操作流程（原子级）：
1. Thread 计算 hash = row_idx % 2048 → shared memory address
2. Thread 发出 atomicCAS(&H.key[hash], EMPTY, row_idx) 抢占 slot
3. 硬件处理：SM 的 LSU (Load-Store Unit) 锁定 shared memory bank
   → 比较 key[hash] 与 EMPTY → 相等则写入 row_idx → 返回旧值
4. 若 atomicCAS 成功（return EMPTY or row_idx）：atomicAdd on H.val[hash]
5. 若失败：next_hash = (hash + STRIDE) % 2048 → 重复步骤 2-4
```

Shared memory 的 bank 结构影响 hash table 性能：32 banks × 4B，若多个线程的 hash 映射到同一 bank 的不同地址无冲突，映射到同一 bank 的同一地址产生 bank conflict（串行化）。Modulo hash 保留 row index 低位可能导致 bank conflict；stride probing 可缓解。

与 L1 cache 的对比：若依赖 L1 cache implicit caching（如 atomic write-back 直接 global atomicAdd），cache line 粒度 128B，uncoalesced scatter 导致大量 cache line 浪费（仅使用 4-8B 有效数据）→ L1 hit rate 低 → 频繁 global memory access。Shared memory hash table 作为显式 scratchpad，programmer 精确控制数据 layout 和替换策略。

术语一般如何实现？如何使用？

GPU shared-memory hash table 的实现关键：
- **Hash 函数**: modulo hash (快速，适合 GPU) 或 multiplicative hash (分布更均匀但计算稍重)
- **Collision resolution**: linear probing（cache-friendly）、quadratic probing（减少 clustering）、double hashing（更均匀但多一次 hash 计算）
- **Concurrency**: atomicCAS on 64-bit packed slot（key+value 打包）实现 single-instruction lock-free insertion；或 warp-cooperative protocols（如 ballot + shuffle 减少 atomic 次数）
- **Load factor**: 通常 <50% 以控制 probing 链长度
- NVIDIA cuCollections 库提供 GPU hash map 标准实现，支持 linear probing + atomicCAS + cooperative groups，H100 上 87.5 GB/s insert throughput
- WarpCore 库支持 AOS layout 64-bit CAS，GV100 上 1.6B inserts/sec
- 关键约束：shared memory 容量有限（A100 168KB/SM, H100 228KB/SM），hash table size 必须与 occupancy 平衡

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

