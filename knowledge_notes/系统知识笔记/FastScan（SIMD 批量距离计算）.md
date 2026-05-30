## FastScan（SIMD 批量距离计算）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FastScan 是一种基于 SIMD 指令集的批量距离计算方法，由 André et al. 在 VLDB 2016 提出（"Cache Locality is Not Enough"）。核心思想：将查表式距离计算转换为 SIMD 寄存器内批量计算。对于量化码与查询向量之间的内积，FastScan 将多个量化码打包到 SIMD 寄存器中（通常 32 个 8-bit 码或 16 个 16-bit 码），执行单指令多数据（SIMD）乘加操作一次处理多个向量。相比传统 PQ 的 RAM 查表（L1 cache latency ~4 cycles），SIMD 寄存器操作仅需 1 cycle，且并行处理多个向量，获得 4-6× 加速。在 Extended RaBitQ 中，FastScan 用于批量计算 ⟨ȳ₀,q'⟩（ȳ₀ 为 1-bit 二进制码），利用 AVX512 指令集。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# FastScan 在 Extended RaBitQ 两阶段距离比较中的角色

# Stage 1: FastScan 批量计算 MSB 距离（剪枝阶段）
# ȳ₀ ∈ {0,1}^D, 32 个向量的 MSB 打包到一个 256-bit SIMD 寄存器
# 使用 AVX512: _mm512_dpbusd_epi32（点积指令）

def fastscan_compute(vectors_msb[32][D], q'[D]):
    # 将 32 个 D-bit 二进制码按 bit 维度重组
    # 每个 32-bit word 包含 32 个向量同一维度的 MSB
    for chunk in 0..D/32:
        v = load_32vectors_msb_at_chunk(chunk)
        r = _mm512_maddubs_epi16(v, q'_chunk)  # SIMD 乘加
        # 累加到结果寄存器
    return dist_estimates[32]

# 剪枝逻辑
for each 32-vector batch:
    dists = fastscan_compute(batch, q')
    for i in 0..31:
        if dists[i] + error_bound < best_dist:
            mark_for_stage2(batch_idx * 32 + i)

# Stage 2: 仅对未剪枝的候选计算完整距离
# ȳ_last 为 (B-1)-bit 整数，复用 SQ 的 SIMD 实现
```

在 Extended RaBitQ 中，FastScan 的关键作用：(1) MSB (ȳ₀) 恰好是 RaBitQ 二进制码，可直接用 FastScan 批量计算；(2) 粗估距离配合理论误差界可安全剪枝；(3) 只有少数候选需要访问 ȳ_last 做精确计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FastScan 的实现：(1) PQ Fast Scan（VLDB 2016）——将查表替换为 SIMD 寄存器内查找；(2) Quicker ADC（TPAMI 2019）——进一步优化量化码布局以适配 SIMD 宽度；(3) Faiss 中的 `IndexIVFFastScan` 和 `IndexIVFAdditiveQuantizerFastScan`——原生支持 4-bit PQ 的 FastScan；(4) SymphonyQG (SIGMOD 2025)——将 FastScan 与 RaBitQ 和 SOTA 图索引结合。实现要求：(a) 批量大小对齐 SIMD 寄存器宽度（如 32 个 8-bit 值）；(b) 向量维度补齐到 SIMD 宽度倍数；(c) 使用特定 SIMD 指令（SSE4/AVX2/AVX512/NEON）。对本论文场景，FastScan 使 Extended RaBitQ 在同样 recall 下比 LVQ 显著更高效。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search
