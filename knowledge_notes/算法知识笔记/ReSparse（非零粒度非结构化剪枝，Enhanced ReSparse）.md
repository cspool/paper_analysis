## ReSparse（非零粒度非结构化剪枝，Enhanced ReSparse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ReSparse 是 AccelES（HPCA 2025，同组先前工作）提出的稀疏 embedding 检索非结构化剪枝算法：在非零元素（non-zero）粒度上，把幅度小于阈值的小非零元素置零，以减少冗余计算与访存。其核心直觉：稀疏向量中小幅度非零对相似度排序影响有限（Retrieval 场景下剪掉它们几乎不损失 Recall）。ParetoES 首次把非结构化剪枝应用于"选择性计算"Top-K SpMV 范式（此前仅用于全计算范式），并做两处结构性增强：(1) 剪枝阈值计算改为"仅对非零元素取均值"——原始 ReSparse 用全局均值作阈值，被大量零元素拉低，导致高稀疏矩阵下大部分非零被保留、剪枝几乎失效（Sp.10M 密度 0.72% 时剪枝率仅 2.34%）；改为非零均值后剪枝率恢复（Sp.10M 达 11.48%）；(2) 剪枝比例计算与 Spherical K-means++ Refine 迭代集成，并在 Recall@100≥80% 约束下确定每数据集最小 nprobe 以保证精度。效果：增强版 ReSparse 峰值剪枝率 61.25%、平均 37.41%（原始 23.93%），相对 AccelES 平均再减非零 18.09%（最高 39.14%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
剪枝伪代码（增强版，以簇为单位离线执行）：
```
for each cluster c:
    nz = nonzeros(submat[c])                       # 簇内非零元素集合
    thr_c = alpha * mean(|v| for v in nz)          # 仅非零均值，alpha 为比例系数
    for (i, j, v) in nz:
        if |v| < thr_c:  submat[c][i,j] = 0        # 幅度低于阈值 -> 剪掉
    prune_ratio[c] = pruned_nnz / total_nnz
# 与 Refine 集成：在 Recall@100>=0.8 约束下迭代选 nprobe，
# 使剪枝后的簇子矩阵仍满足目标召回
```
张量计算例子：某簇含 1000 个非零、均值为 3.2，阈值取 0.5×3.2=1.6，则幅度 <1.6 的非零（如 0.3、1.1）置零，后续 SpMV 跳过这些位置。对 FPGA 而言剪掉的非零直接减少 HBM 流式读取字节与 DSP 乘加次数（sparse-agnostic 架构下剪枝收益与稀疏度成正比）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AccelES 在 Ultra-CSR/Random-CSR 编码前离线剪枝，配合低比特量化（INT6）实现 73.5% 平均访存减少与 2.7× 计算并行度（HPCA 2025 数据）；ParetoES 把它集成进聚类-量化-剪枝-编码预处理流水（A100 GPU 上执行，全精度聚类后、INT6 量化后剪枝），剪枝后矩阵按簇重排为子矩阵、编码 Ultra-CSR 载入 FPGA HBM。使用注意：剪枝率需按数据集/Recall 目标标定，过度剪枝会掉 Recall（论文在 Recall@100≥80% 约束下选最小 nprobe）；对非 sparse-agnostic 的 CPU/GPU 平台，剪枝收益难以兑现（Faiss+ReSparse 在 CPU 仅 +14%、GPU 反而 -7%）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
