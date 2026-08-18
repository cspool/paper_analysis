## 非均匀量化与学习码本（NU-WAQ / K-Means 码本量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
非均匀量化（non-uniform quantization，NU-WAQ 指权重-激活双侧非均匀量化）用不等间隔的量化电平（centroids/质心）拟合数据真实分布，与均匀量化（固定 scale+zero-point 的等间隔整数映射）相对。学习码本（learned-codebook）方法通过聚类/训练优化质心，代表即 K-Means 量化（论文公式1）：x̃_i = C_{idx_i}，idx_i = argmin_k ‖x_i − C_k‖²——即用 n-bit 索引矩阵 + 2^n 个 FP 质心码本表示数据，重建靠查表。因为质心可贴合 LLM 权重/激活的重尾+离群分布，NU-WAQ 在低比特下精度显著优于均匀方案（SqueezeLLM 3-bit LLaMA-7B PPL 6.32 vs GPTQ 7.55）。OASIS（§III-A）具体化：权重 4-bit 采用输出通道级量化（整矩阵共享质心 + 每输出通道独立缩放因子，无 outlier 保护）；激活 3/4-bit 采用 token 级量化（每 token 独立质心与缩放因子），激活质心用 C4 数据集 16 个校准样本、经 Fisher 信息矩阵加权的 K-Means 离线学习，在线只做聚类分配（offline/online 质心 RMSE 仅 0.01，图5，验证离线学习可行性）。其余参考：低精度浮点格式（MXFP4、NVFP4）也属非均匀量化但非学习码本。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线量化（OASIS §III-A）
W_c, W_idx = kmeans(W, k=2^4)          # 4-bit 权重：16 质心 + 索引矩阵
A_c = fisher_weighted_kmeans(calib_acts(C4, 16 samples), k=2^4)  # 激活质心
# 在线推理（每 token）
idx = argmin_k ||x - A_c[k]||^2         # 聚类分配（OASIS 用 Clustering Unit 硬件）
x̃ = A_c[idx]                            # 重建（查码本）
```
压缩比：n-bit 时索引矩阵 K×N×n bit + 码本 K×2^n×16 bit（权重）或每 token 码本；token 级激活量化把量化参数动态化——这是其精度高但需在线聚类开销的来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表开源实现：SqueezeLLM（github.com/SqueezeAILab/SqueezeLLM，3/4-bit K-Means + dense-sparse 分解）、Any-Precision LLM（多精度分裂质心）、Bitsandbytes（NF4）、SpQR（稀疏+非均匀混合）。OASIS 本身无公开代码（arXiv:2507.23035 无 Code 链接）。使用场景：超低位宽（≤4-bit）权重/激活压缩；注意 NU-WAQ 索引格式与现有 INT 低精度计算单元不兼容——传统执行需反量化为 FP16 再 GEMM，OASIS 用 LUT-GEMM 直接计算（见 WAQ LUT-GEMM 条目）。

涉及论文标题：
- OASIS Outlier-Aware LUT-Based GEMM with Dual-Side Quantization for LLM Inference Acceleration
