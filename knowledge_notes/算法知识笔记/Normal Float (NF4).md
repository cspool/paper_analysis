## Normal Float (NF4)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Normal Float (NF4) 是由 Dettmers et al. (2023) 在 QLoRA 工作中提出的 4-bit 查找表量化数据类型。NF4 假设网络权重服从正态分布 N(0,σ²)，基于信息论最优原则——每个量化层级应等概率使用（即每个层级映射到相同数量的权重值）。NF4 使用 Gaussian 分位数函数将概率质量均匀分为 16 份，经反函数映射生成 16 个量化值，确保量化后直方图近似平坦。NF4 固定 3 个值：-1、0、1（分别对应最小、零点和最大值），其余 13 个值由 Gaussian 分位数确定。NF4 的值分布特点是中心密集（零附近层级间距小）边缘稀疏（远离零的层级间距大），这匹配了正态分布的概率密度结构。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NF4 的导出基于 Gaussian 分位数函数 Φ⁻¹(p)：

```
# === NF4 码本导出 ===
# 将 (0, 1] 概率空间等分 16 个区间
p_i = i/16 for i=1..16  # 或按 QLoRA 的精确方法
# 保留 -1, 0, 1 为固定值
# 其余值通过 Gaussian 分位数映射
nf4_values = Φ⁻¹(p_i)  # standard normal quantile function
# 归一化使端点为 -1 和 1
```

NF4 在 QLoRA 中的使用流程：
```
W_flat = W.reshape(-1)
blocks = W_flat.reshape(B, 64)   # QLoRA 默认 block size 64
for b in 1..B:
    w_max[b] = max(|blocks[b,:]|)
    for i in 1..64:
        x = blocks[b,i] / w_max[b]
        idx = argmin |x - nf4_table|
        Ŵ[b,i] = w_max[b] * nf4_table[idx]
```

论文指出 NF4 的核心缺陷：正态分布假设不正确。30+ DNN 的 profiling 显示（Table 1, Kolmogorov-Smirnov 检验），大多数 DNN 分布最优近似于 Student's t-distribution（ν≈5），而非正态分布。正态分布无法同时拟合分布的尖峰（peak）和厚尾（tail）。这导致 NF4 的概率分配与真实权重分布不匹配，尤其在分布的尖峰区域欠量化而在尾部过量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NF4 是 QLoRA 量化微调框架的核心组件，通过 bitsandbytes 库（https://github.com/TimDettmers/bitsandbytes）实现。bitsandbytes 提供 CUDA kernel 实现 NF4 的查表解码 → FP16 GEMM。NF4 在推理时完全通过查表操作完成反量化，无算术计算。由于 NF4 需要浮点查找表和高精度累加器，其硬件直接实现成本较高，因此在实际系统中，NF4 更适合作为软件层的量化方案或作为设计硬件高效数据类型（如 E2M1）的精度参考。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
