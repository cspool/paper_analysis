## Approximate Nearest Neighbor (ANN) Search（近似最近邻搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ANN 是 NN 搜索的松弛版：给定 N 个 D 维数据向量和查询 q，返回与 q 距离接近 NN 的向量，允许精度折损换时间/空间节省。指标：recall@K（前 K 中真 NN 比例）、average distance ratio、QPS（效率）。ANN 是向量数据库、RAG、推荐系统、信息检索的核心操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IVF + 量化的 ANN pipeline：
1. Index: KMeans 聚类 → 每聚类中心化 → 量化编码 → 存储压缩码
2. Query: 找 nprobe 最近聚类 → 扫描候选，用压缩码估计距离 → 返回最小估计距离的向量
关键参数：聚类数 L、nprobe（扫描聚类数）、B（量化 bits/dim）。本论文目标场景：仅存储压缩向量（无原始向量），通过中等压缩率独立产生 >95% recall 无需 re-ranking。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流库：Faiss (Meta)、Milvus、ScaNN (Google)、Annoy (Spotify)、NGT (Yahoo)、pgvector。量化+IVF 或图索引是主流方案。本论文方法可在所有 ANN 任务中无缝替换 SQ/LVQ。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
