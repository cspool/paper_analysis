## Non-Uniform Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Non-Uniform Quantization（非均匀量化）是将浮点值映射到任意分布的离散质心（而非等间隔固定点）的量化方法。与均匀量化（uniform quantization）的等间隔 grid 不同，非均匀量化存储一个 codebook C = {c_1, c_2, ..., c_k}，其中质心值可以位于任意浮点位置。每个原始值 x 存储的是映射质心的索引 j（⌈log₂ k⌉ bits），而非量化值本身。解码时通过索引查表恢复近似值。Vector Quantization 是非均匀量化在高维的推广——当 d=1 时即为标量非均匀量化（1D VQ），当 d>1 时将 d 个值作为一个向量整体映射到 d 维质心。

GPTVQ 证实非均匀量化在 low-bitwidth（2-3 bits）下比均匀量化有显著优势：Llama-v2-70B W2@g128，GPTVQ 1D（标量非均匀）PPL=5.03，远低于 OmniQuant（均匀）的 6.55。2D VQ 进一步降至 4.72。这是因为非均匀质心可以密集分布于高概率区域、稀疏分布于尾部，比均匀 grid 更好地匹配 LLM 权重的实际分布（近似高斯/拉普拉斯，而非均匀）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
非均匀量化的编码/解码公式：
```
编码（量化）:
j* = argmin_j |x - c_j|  # 找最近质心（1D）或加权最近质心（VQ）
store(j*)                # 存储 ⌈log₂(k)⌉ bits

解码（反量化）:
x̂ = c_{j*}              # 查表恢复
```

与均匀量化的对比：
```
# 均匀量化 (INT4, 16 个等间隔点)
Δ = max(|w|) / 7
x̂_uniform = Δ · clamp(round(x/Δ), -8, 7)  # 16 个候选值

# 非均匀量化 (1D VQ, k=8 个质心, 3 bits)
C = {c_0, c_1, ..., c_7}  # 可位于任意位置
x̂_nonuniform = C[argmin_j |x - c_j|]      # 8 个候选值但位置灵活
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
非均匀量化的代表性实现：GPTVQ（1D/2D/4D VQ via EM + GPTQ 框架）；AQLM（8D VQ via beam search + block FT）；NF4/NF3（NormalFloat，基于正态分布分位数的固定 codebook）；AFPQ（非对称 FP 量化，正负值各有独立 scale 的 FP 格式）。非均匀量化在 low-bit（<4 bits）下优势最大；在 4-bit 及以上，均匀量化的等间隔 grid 已足够密集，非均匀的边际增益减小。局限：需要 codebook 存储（额外 bits），需要查表解码（增加延迟），不总是适合硬件加速（GPU Tensor Core 偏好均匀 INT 格式）。

GuidedQuant 提出 LNQ (Layer-wise Non-uniform Quantization)，一种交替优化算法替代 GPTVQ 1D：codebook 用闭式最小二乘解（c = (PᵀHP)⁻¹PᵀHw_j）替代梯度下降，assignment 用 cyclic coordinate descent 替代 GPTQ，保证目标函数单调递减并收敛。LNQ 在所有 settings 下优于 GPTVQ 1D，可与 GuidedQuant 的 end loss guided 目标结合进一步提升性能。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- SqueezeLLM Dense-and-Sparse Quantization
