## ANNS（Approximate Nearest Neighbor Search，近似最近邻搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANNS 是最近邻（NN）搜索的松弛版本：给定 N 个 D 维数据库向量与查询向量 q，返回与 q 距离最接近的 k 个向量的近似结果，以可接受的精度损失换取亚线性搜索时间。核心距离度量包括 L2 范数与内积（IP）。精度用 recall@k = |P'∩P|/|P|（ANNS 返回集与真 kNN 集的重合比例）衡量，效率用 QPS（每秒查询数）衡量。ANNS 是向量数据库、RAG（检索增强生成）、推荐与信息检索的核心算子；本论文将其置于 LLM RAG 场景，检索阶段的内存带宽瓶颈直接决定整体推理性能。索引方法分为哈希式、树式、量化式与图式四类，其中图式（graph-based）在商用数据库（Milvus、Weaviate 等）与 RAG 系统中被广泛采用，可提供数量级的吞吐提升同时保持高 recall（vault 笔记：/data3/paper_analysis/knowledge_notes/算法知识笔记/Approximate Nearest Neighbor (ANN) Search（近似最近邻搜索）.md）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 ANNS 查询 pipeline：① 索引构建（一次性离线）——把语料向量组织成可搜索结构（如 HNSW 多层图、IVF 聚类+量化码）；② 查询搜索（在线、反复执行，决定系统性能）——从入口出发遍历索引，对候选向量计算与 q 的距离并维护 top-k 候选。对图式 ANNS（HNSW），查询 pipeline 为逐层 BFS：初始化候选优先队列（含入口点）→ 每 hop 取出队列最近点 → 取其邻居表 → 计算邻居与 q 的距离（全 D 维）→ 距离小于 threshold（队列最远点距离）则插入队列 → 重复直到队列耗尽。伪代码（HNSW 搜索，每 hop）：
```
cand ← {entry}; visited ← ∅; threshold ← +∞
while cand 非空:
  x ← cand 中距 q 最近且未访问的点; visited ← visited ∪ {x}
  for n in neighbors(x):                       # 邻居表查找
    if n ∈ visited: continue
    d ← distance(q, n)                          # 全 D 维距离计算（内存受限热点）
    if d < threshold:
      更新候选队列并弹出最远点; threshold ← 队列最远距离
```
核心特征：距离计算算术强度极低（每元素仅一次减/乘/加），性能完全被数据访问带宽限制（roofline 分析见硬件架构条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现库：FAISS（Meta）、ScaNN（Google，SCANN 为本论文 CPU SOTA baseline）、hnswlib、Milvus、cuVS（NVIDIA，本论文用其构建 HNSW 索引）。本论文：以 HNSW 图式 ANNS 为对象（Dfloat 数据布局 + FEE-sPCA 早退 + NDP 加速），baseline 含 CPU 的 HNSW/SCANN、GPU 的 CAGRA、NDP 的 ANSMET 等，在 SIFT/GIST/BigANN/GloVe/Wiki/MS_MARCO 六数据集上以 recall@k≥90% 比较 QPS。开源：https://github.com/Intelligent-Computing-Research-Group/NasZip。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
