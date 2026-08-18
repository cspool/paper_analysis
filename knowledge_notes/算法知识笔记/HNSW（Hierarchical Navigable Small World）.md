## HNSW（Hierarchical Navigable Small World）

术语解释
- HNSW 是多层可导航小世界图 ANN 索引：层 0 含全部节点、上层节点指数级稀疏、层间长程连接，查询从顶层粗粒度贪心下降到底层精化，搜索复杂度 O(log n)；论文在 SSD-resident ANN 案例中用它并配合"图链接元数据与节点同驻 SSD、高层热节点驻 DRAM"的放置策略。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HNSW（Malkov & Yashunin 2018）把向量组织成多层邻近图：层 0 最密（含所有节点），越往上节点越少、连接越长程，节点所在最高层由指数衰减概率决定（mL 参数）；查询从顶层入口贪婪导航逐层下降，每层局部贪心搜索候选集，复杂度 O(log n)（网络来源：Milvus/Pinecone/Weaviate/FAISS/cuVS 等向量库的骨干）。论文利用 HNSW 的层次访问模式：高层节点少、访问间隔短（DRAM 友好），低层节点多、访问间隔长（SSD 友好），把图链接元数据与节点 co-locate 在 SSD、DRAM 只缓存高层节点；用校准的 layer-aware 合成 trace 模拟其 coarse-to-fine 流水。
- 从算法pipeline角度拆解术语：一次查询的 pipeline：从顶层入口开始 → 逐层贪婪下降（每层访问若干节点，比较距离）→ 在层 0 找到 k 近邻候选。SSD-resident 版本：每访问一个节点需从 SSD 读其向量与邻居链接（小块随机读）；高层节点命中 DRAM，低层节点读 SSD。论文叠加两阶段渐进（见"两阶段渐进式 ANN"条目）：先用 512B reduced 向量粗筛、再对 5%-20% promoted 候选取 full 向量精排，使大部分访问落在高 IOPS 的小块读上。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现参数（网络来源）：M（每节点连接数，默认 16）、m_max0（层 0 上限 2M）、ef_construction（建图候选数 200）、ef_search（搜索候选数 50）、mL（层分配衰减）；开源实现包括 FAISS、hnswlib、Milvus 等。论文使用方式：80 亿 embedding 语料、reduced 固定 512B、full 2/4/6/8KB（promotion 5%/10%/15%/20%），GPU+SN 达 13-17 KQPS（512GB DRAM），相对 Normal SSD 一致 2-3×；对照 DiskANN（约 5 KQPS 量级）。论文为模型驱动评估，无开源实现。
- NasZip 补充视角（ISCA'26，HNSW 作为 NDP 加速对象）：NASZIP 聚焦 HNSW 搜索阶段（反复执行、主导系统性能），在其上叠加 FEE-sPCA 早退（PCA 变换后按 burst 估计距离提前剪枝）、Dfloat 位级压缩、DaM 邻居表映射与 LNC 缓存/预取，并给出 NDP 执行模型：候选优先队列维护 threshold（队列最远点距离）、逐 hop BFS 由 NDP 的 VPE 并行算距、邻居表查找卸载到 NDP 并按数据感知映射避免跨 sub-channel 通信。efSearch 增大提升 recall 但降 QPS（Fig.19），batch=16 为吞吐/延迟折中。HNSW 索引用 NVIDIA cuVS 构建（预建索引保证可复现，自建因随机性略有偏差）。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
