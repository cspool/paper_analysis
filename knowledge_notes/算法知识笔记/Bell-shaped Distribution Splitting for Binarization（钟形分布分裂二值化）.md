## Bell-shaped Distribution Splitting for Binarization（钟形分布分裂二值化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bell-shaped Distribution Splitting for Binarization（钟形分布分裂二值化）是 BiLLM（ICML 2024）提出的针对非 salient 权重的二值化策略。LLM 的权重（去除少数 salient 权重后）呈对称钟形分布（类似高斯或拉普拉斯分布，大多数值聚集在 0 附近）。直接对这些权重进行二值化（均匀量化极端情况，仅 ±α 两个量化级）会产生极大的 MSQE（Mean Squared Quantization Error），因为大量聚集在 0 的值被强制映射到 ±α。BiLLM 的解决策略是：搜索一个最优分裂点（break-point）p*，将钟形分布沿 p/-p 切割为两个区域——集中区（|w| ≤ p，权重密集在 0 附近）和稀疏区（|w| > p，分布在尾部），然后分别以独立的 scaling factor（α_c, α_s）对各区域独立二值化。这相当于用两个分段常数函数逼近钟形分布曲线。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
BiLLM 的分裂搜索和分段二值化流程（以 LLaMA-7B 为例）：
```
W_nonsal = W_{:, not in salient_cols}         # 非 salient 权重
m = max(|W_nonsal|)                            # 权重极值

# 搜索最优 break-point p*（百分位搜索，步长 0.1）
e_best = inf; p_best = 0
for ratio in [0.1, 0.2, ..., 0.9]:
    p = ratio × m
    W_c = {w ∈ W_nonsal : |w| ≤ p}             # 集中区 (concentrated)
    W_s = {w ∈ W_nonsal : |w| > p}             # 稀疏区 (sparse)
    
    # 各自独立二值化（Equation 12）
    α_c = ||W_c||_ℓ1 / n_c; B_c = α_c · sign(W_c)
    α_s = ||W_s||_ℓ1 / n_s; B_s = α_s · sign(W_s)
    
    # 总体 MSQE（Equation 11）
    θ²_p = ||W_s - B_s||² + ||W_c - B_c||²
    
    if θ²_p < e_best: e_best = θ²_p; p_best = p

# 最终二值化
Ŵ_nonsal = B_c(p_best) + B_s(p_best)          # 1-bit 参数 + 1-bit group flag
```
搜索曲线呈凸性（paper Appendix C 验证），保证了全局最优解的存在。额外开销为 1 bit 用于区分 sparse/concentrated 组（不参与 GEMM 计算）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 BiLLM GitHub 仓库中，binary.py 文件的 `seg_search` 函数实现了百分位搜索。该方法适用于任意呈钟形分布的非 salient 权重集合。搜索范围通常限制在 max(|W|) 的 10%-90% 之间。关键设计选择：(1) OBC 块级补偿在分布搜索之前进行，补偿后的权重分布更接近理想高斯，有利于凸性保证；(2) block_size=128 为最佳平衡点（更小 block 精度更高但 flag 位开销增大）；(3) 仅在一个维度使用分裂（不再嵌套多层分裂），以平衡精度和实现复杂度。该方法在 OPT-6.7B 上对非 salient 权重的二值化提升尤为显著（ablation 显示 splitting-only 提升 > residual-only 提升，Figure 8）。

涉及论文标题：
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---
