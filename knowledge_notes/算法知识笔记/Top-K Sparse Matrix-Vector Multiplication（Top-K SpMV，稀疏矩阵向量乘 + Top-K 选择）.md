## Top-K Sparse Matrix-Vector Multiplication（Top-K SpMV，稀疏矩阵向量乘 + Top-K 选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-K SpMV 是稀疏 embedding 相似度检索的核心算子：给定稀疏矩阵 A∈R^(m×n)（每行一个数据库候选向量）与稠密查询向量 v∈R^n，计算 y=A·v 后返回最大 K 个 y 值的索引集合 TopK(y,K)={i_1,...,i_K}（y_ij≥y_r 对所有 j 且 r∉TopK）。在检索系统中 A 的行是稀疏 embedding（RAG、推荐、知识图谱中的实体/文档/商品向量），v 是查询，内积即相似度。问题同时含两部分：稀疏矩阵-向量乘（SpMV）与 Top-K 选择。工作量三大特征（Fig.2）：(1) 不规则访存——稀疏矩阵访问随机、缓存命中率低、预取无效，CPU 上内存延迟占执行时间 60–70%；(2) 计算负载不均——各行非零数分布不均导致并行核间失衡；(3) 稀疏输出——K≪m，需堆或预过滤结构在线维护 Top-K。近似精度用 Recall@K=|TopK_approx∩TopK_exact|/K 衡量（TopK_exact 为全精度穷尽 Top-K SpMV 结果）。论文指出处理约 20% 候选向量即可达 ~80% Recall（Pareto 规律），启发"选择性计算"范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 Top-K SpMV 检索的算法 pipeline（ParetoES 的选择性计算版本，区别于全计算）：
```
# 离线：聚类索引构建（A 行按簇重排、量化、剪枝、编码）
clusters = SphericalKmeansPPRefine(A, K=nlist)      # 聚类 + 动态精化
A_q = INT6_quantize(A)                               # 对称 6-bit 量化
A_q = ReSparse_prune(A_q)                            # 非零粒度剪枝
submat = partition_by_cluster(A_q)                   # 按簇 ID 重排为子矩阵
# 在线：单查询 v
v_q = INT6_quantize(v)
nprobe = lookup_min_nprobe(recall_target=0.8)        # 满足 Recall@100≥0.8 的最少簇数
top_probe = topk({<mu_c, v_q> for c in clusters}, nprobe)   # 质心相似度选簇
y = 0
for c in top_probe:
    for (i, j, val) in submat[c]:  y[i] += val * v_q[j]    # 簇内稀疏 SpMV
result = topk(y, K=16) per core -> aggregate top-512      # 核内 Top-16 聚合
```
张量计算例子：对选中簇子矩阵，y_i=⟨A_i,v⟩=Σ_{j∈nnz(A_i)} A_i[j]·v[j]，仅在非零元素上做乘加（INT6×INT6），跳过全零列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPU/GPU 通用实现：MKL 的 mkl_sparse_?_mv + sort、sparse_dot_topn（ing-bank，Python 稀疏矩阵乘 + top-n 选择，精确全计算基线）、cuSPARSE（SpMV kernel）+ Thrust（sort-select pipeline）、Faiss（IndexIVFFlat 倒排索引 + nprobe 可调，支持选择性计算但基于 K-means 且非稀疏无关）。FPGA 实现：FPGA32（BS-CSR 块流式）、AccelES（Ultra-CSR/Random-CSR + 低比特 + ReSparse）、ParetoES（ACPE 多核 + DMSU Bitonic-16 微排序）。ParetoES 实验：Recall@100∈[0.8,1.0] 下 QPS 最高 4761.9（Sp.Baidu），比 CPU/GPU baseline 高至 540×/79×，平均 2.27× vs AccelES。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
