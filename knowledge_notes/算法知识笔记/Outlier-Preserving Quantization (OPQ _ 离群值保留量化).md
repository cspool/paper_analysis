## Outlier-Preserving Quantization (OPQ / 离群值保留量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OPQ（Outlier-Preserving Quantization）是 BOF4 论文提出的混合精度 block-wise 量化策略。解决 outlier weights 破坏 block-wise absmax 归一化分布假设的问题：少数极端 outlier 导致其 block 的 `w_b^max` 异常大，使归一化后非 outlier 权重被过度压缩到零附近（underrange），量化器偏离最优设计区间。OPQ 在量化前将 outlier 替换为 0，单独存储为 BF16 + 64-bit position index。Outlier 判定：`|w_{b,i}| > σ_b * F_M^{-1}(q)`，其中 σ_b 为 block 样本标准差（Eq. 73），`F_M^{-1}(q)` 为绝对 block maxima 分布的 q-分位数（q=0.95）。OPQ 与任意 block-wise 量化方法组合，码本不变。额外内存：I=64, q=0.95 时约 0.96%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# OPQ Algorithm (q = 0.95)
threshold = F_M^{-1}(0.95)  # quantile of absolute block maxima distribution

# Step 1: Per-block standard deviation
for each block b:
    σ_b = std(block[b, :])

# Step 2: Outlier detection and removal
outlier_store = []
for each weight w[b,i]:
    if abs(w[b,i]) > σ_b * threshold:
        outlier_store.append((position, w[b,i]))
        w[b,i] = 0  # replace with zero

# Step 3: Standard block-wise quantization on cleaned weights
w_quant, w_max = blockwise_absmax_quantize(w, codebook, I)

# Step 4: Decode — reconstruct from 4-bit, then overwrite outlier positions with BF16
```
Outlier 判定直观理解（Fig. 7）：`σ_b * F_M^{-1}(0.95)` 表示"95% block 中绝对最大值都不超过的阈值"。超过该阈值说明该权重大得不正常（非高斯），应作 outlier 处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/ifnspaml/bof4。OPQ 与具体码本解耦（可与 NF4, AF4, BOF4, BOF4-S 任意组合）。运行时开销极小（RTX 4070 Ti Super 上生成 1000 tokens 额外耗时随 I 增大递减，Fig. 11）。大 block size（I ≥ 128）效果尤其显著。推荐 q=0.95（{0.9, 0.95, 0.97, 0.99} 中最佳平衡）。

SpQR 提出另一种 outlier 处理方法：使用 OBS 框架的封闭形式敏感度准则 s_ij = (w_ij − quant(w_ij))² / (2[H⁻¹]_jj) 而非 σ_b 乘以分位数阈值。SpQR 的 outlier 判定发生在量化过程中（而非预处理），通过 leave-one-out error 对比动态确定：E_base − E_ol > τ。Outlier 以 CSR 稀疏格式存储（32 bits/outlier：16-bit value + 16-bit col index），约 1% 的权重被保留为 16-bit。非 outlier 权重以 3-4 bit 量化，排除 outlier 后 min-max scale 显著减小。

涉及论文标题：
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

---
