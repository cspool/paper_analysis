## LUT-GEMM (Lookup Table based GEMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LUT-GEMM 是一种利用查找表（Lookup Table）加速二值/低比特量化权重 GEMM 的 GPU kernel 优化方法。核心思想：当权重矩阵是二值（{-1,+1}）时，与激活向量的内积退化为对激活元素的加/减操作。将权重按每 μ 个比特分组（通常 μ=3~4），预计算这 2^μ 种二值模式与激活片段的所有可能内积结果存入 LUT。实际计算时，用 μ-bit 权重模式作为索引直接查表获取结果，避免了逐个浮点乘加。NAVER 的 LUT-GEMM (Park et al., 2022) 首次将 BCQ 格式下的 LUT 计算实现在 GPU 上，支持 uniform 和 non-uniform (BCQ) 两种量化方案。后续工作：FLUTE (MIT/CMU, EMNLP 2024) 通过 LUT 向量化和跨 shared memory bank 复制消除 bank conflicts，实现 2-4× GEMM 加速；LUT Tensor Core (Microsoft, ISCA 2025) 提出 dedicated LUT-based Tensor Core 硬件设计；FIGLUT (POSTECH+NAVER, HPCA 2025) 设计 custom RAC（Read-Accumulate）单元替代 MAC。在 AnyBCQ 中，LUT-GEMM 思想被用于自研 CUDA kernel：每个比特平面 B_i ∈ {-1,+1} 的 GEMM 通过 LUT 加速加减操作，p 个比特平面结果乘以 α_i 后累加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LUT-GEMM 的基本伪代码（以 μ=4 为例）：

```
# 输入: activation A ∈ R^{1×K}, binary weight B_i ∈ {-1,+1}^{K×N}, μ=4
# 每 μ 个连续 K 维度为一组，构建 2^μ=16 entry LUT

for n in range(0, N, TILE_N):  # 输出 tile
    for k_tile in range(0, K, TILE_K):  # 输入 tile
        # Step 1: 构建 LUT (per tile)
        A_tile = A[k_tile : k_tile+TILE_K]  # 激活片段
        LUT = [0] * (1 << μ)  # 2^μ entries
        for g in range(0, TILE_K, μ):  # 每 μ 个元素一组
            for pattern in range(1 << μ):
                # pattern 的 μ bits 决定加减模式
                sum_val = 0
                for bit in range(μ):
                    sign = 1 if (pattern >> bit) & 1 else -1
                    sum_val += sign * A_tile[g + bit]
                LUT[pattern] += sum_val
        
        # Step 2: 查表获取 partial sums
        for n_tile in range(0, N, TILE_N):
            for g in range(0, TILE_K, μ):
                w_bits = B_i[k_tile+g : k_tile+g+μ, n_tile]  # μ-bit pattern
                output[n_tile] += LUT[w_bits]  # 直接查表！
```

GPU 上的关键瓶颈：多个线程同时访问 LUT 的不同 entry 时产生 shared memory bank conflicts。解决方案：(1) FLUTE 的 LUT 跨 bank 复制（每 bank 存完整 LUT）；(2) T-MAC 的 in-register table（使用 ARM TBL / x86 PSHUF 指令避免 shared memory）；(3) FIGLUT 的 custom decoding 硬件消除 bank conflicts。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LUT-GEMM 的开源实现：(1) github.com/naver-aics/lut-gemm：NAVER 的官方实现，支持 BCQ + uniform 量化；(2) FLUTE (MIT)：CUDA kernel library，2-4× GEMM 加速；(3) T-MAC (Microsoft)：CPU 端实现，in-register LUT。在 AnyBCQ kernel 中，LUT-GEMM 用于每个比特平面的 GEMM 加速，p 个平面的 LUT-GEMM 结果按 α_i 加权累加。GPU 实现要点：LUT 大小 = 2^μ × sizeof(FP16)，μ=4 时 32 bytes，适合 constant memory 或寄存器；μ 越大查表次数越少但 LUT 越大（μ=5: 64 bytes, μ=6: 128 bytes）。权衡：LUT-GEMM 在 memory-bound 场景（decode, batch=1）收益最大，因减少的算术操作等价于更低的 arithmetic intensity 要求。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---
