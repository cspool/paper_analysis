## Hybrid SME/SVE Matrix-Vector Co-execution

术语是什么？

Hybrid SME/SVE Matrix-Vector Co-execution 是 ASM-SpMM 提出的混合执行策略：将稀疏矩阵中密度足够高的 block 分配给 SME outer-product path 以获得高吞吐，将低密度/碎片化 block 分配给 SVE/Neon vector path 以避免 SME 在低利用率 block 上空转。关键是 interleaved instruction scheduling——vector path 的计算被安排在 SME path 的固定执行窗口内（如 ZA tile 切换间隙或 prefetch 等待周期），使 vector 工作量被 SME 延迟隐藏。

从 kernel 调度角度拆解术语：

Hybrid kernel 的调度决策和执行流：
```
// Phase 1: Block partitioning 决策
for each row_window_r:
    density = nnz_in_window / (SVL * num_cols_in_window)
    if density < THRESHOLD:
        assign_to_vector_path(r)    // SVE/Neon path
    else:
        assign_to_sme_path(r)       // SME outer-product path

// Phase 2: Interleaved execution schedule
sme_blocks  = sorted_by_density_desc(sme_path_blocks)
vec_blocks  = sorted_by_density_asc(vector_path_blocks)

// 交错执行：SME path 执行一个 block → vector path 执行若干 block
while sme_blocks or vec_blocks:
    if sme_blocks:
        block = sme_blocks.pop()
        ZA_tile = sme_outer_product_spmm(block)      // SME 高延迟(10-20 cycles)
        // vector path 在 SME 执行期间并行执行
        while estimated_vec_time < sme_latency:
            if vec_blocks:
                vec_block = vec_blocks.pop()
                vec_result = sve_vector_spmm(vec_block)  // SVE/NEON low latency
    // 合并 SME 和 vector path 结果
    merge_to_output(ZA_tile, vec_result)
```

关键设计考量：
1. **SME/SVE microbenchmark 延迟估计**：预先 profile 不同密度 block 的 SME 和 SVE 执行时间，运行时用于 decide block allocation 和 interleave granularity
2. **Interleaved scheduling**：要求在 SME 固定执行窗口内完成 vector 工作量——vector block 太小则调度开销淹收益，太大则超越 SME 窗口产生额外等待
3. **Hybrid/theory ratio**：论文报告 hybrid 实际性能/理论性能为 0.78-0.90——差距来自 register partition（ZA 和 Z register 划分给 SME/vector path）、资源竞争（SVE 和 SME 共享 load/store 带宽）和 workload partition 开销
4. **收益场景**：在 rCA、FY-RSR、ddi、ppi 等包含大量低密度 block 的矩阵上，hybrid kernel 比 matrix-only (纯 SME path) 快 8%-18%

术语一般如何实现？如何使用？

实现需同时支持 ARM SME intrinsics 和 SVE/NEON intrinsics，在同一个 kernel function 内做条件分支和 instruction interleaving。Apple M4 的 E-core 上 SME 性能极低（约 P-core 1/8-1/16），hybrid path 的 SVE/NEON fallback 在 E-core 上尤为重要。LOOPS（同期工作，arxiv 2511.08158）采用了类似的 hybrid 思路：row-wise CSR 分配给 NEON + vector-wise BCSR 分配给 SME，在 M4 Pro 上达 9.93× (FP32) / 14.4× (FP64) over TACO。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

