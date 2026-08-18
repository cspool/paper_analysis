## 选择性计算与簇探测（Selective Computation / Cluster Probing with nprobe）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
选择性计算（selective computation）是 ParetoES 的核心范式：不做全库穷尽 Top-K SpMV，而是先按聚类质心相似度选出 Top-nprobe 个最相关簇，只在这些簇对应的子矩阵上计算相似度，用少量扫描换取 Recall@100∈[0.8,1.0] 的可接受召回。动机（Pareto 规律）：处理约 20% 候选向量即可达 ~80% Recall；生产 RAG/推荐系统在 80–90% Recall 下几乎无损（RAG 掉 Recall 到 80% 仅损 0.3–3.6% 准确率，推荐系统 GMV 从 92.5%→85.5% Recall 仅 -1.2%），而 1s 延迟上升可致收入 -7~10%。对比：Faiss 的 nprobe 也是"扫前 nprobe 个倒排桶"，但基于 K-means（L2）聚类质量差、且 SIMD/warp 架构对稀疏不规则访存低效；FPGA 全计算加速器（AccelES/FPGA32）固定延迟、无法运行时用 Recall 换吞吐。ParetoES 用硬件原语实现选择性计算：ACPE 的 Bitonic-16 质心筛选 + Mem Map 调度器按 sub_nprobe 动态取簇块，nprobe 参数通过软件接口（sub_nprobe=⌈nprobe/32⌉/核）运行时配置、无需重综合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
选择性计算检索流程（单查询）：
```
# 簇筛选（硬件：Bitonic-16 质心排序）
scores_c = [<mu_c, v_q> for c in 0..C-1]          # 与全部质心内积（质心在 HBM 通道头部）
top_probe = topk(scores_c, nprobe)                  # 选出 nprobe 个簇（软件配置）
# 簇内评估（仅扫描选中簇子矩阵）
for core in 0..31:                                  # 32 ACPE 并行
    for t in 0..sub_nprobe-1:                       # 每核承担 sub_nprobe=ceil(nprobe/32) 个簇
        addr = LUT[core][t]                          # Mem Map 查表定位簇块地址
        stream_cluster_block(addr) -> decode -> INT6 MAC -> aggregate
    local_top16 = bitonic16(topk, K=16)
global_top512 = merge(all cores' top16)             # host 聚合
Recall = |top512 ∩ exact_topK| / K
```
张量计算：只在选中簇的稀疏子矩阵上算 y_i=⟨A_i,v⟩（跳过未选中簇的整块矩阵，消除 68%/44%/4% 访存+计算对应 Recall 0.8/0.9/1.0）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现：Faiss IndexIVFFlat 的 nprobe 参数即此概念（倒排桶探测）。ParetoES 硬件实现：ACPE 把簇筛选（Bitonic-16 排序质心分数）与簇内 Top-K（DMSU 局部 Top-16）做成固定流水，sub_nprobe 由 host 在初始化时计算下发（软件-硬件协同设计，避免参数调优触发重综合）；Recall 目标通过查表映射到 nprobe（论文在 Recall@100≥80% 下求每数据集最小 nprobe）。使用时在 Recall 与吞吐间沿 Pareto 前沿调节：Recall@100≈0.8/0.9/1.0 时 Sp.Baidu QPS=4761.9/2857.1/1851.9（nprobe=128/224/384，nlist=398），相比 AccelES 固定 1818.2 QPS（无选择性计算）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
