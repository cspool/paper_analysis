## 结构编码（Structural/Positional Encoding，SE/PE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 结构编码是图 Transformer 为捕获顶点拓扑属性而加入的显式编码，可融入输入特征（如度嵌入）或注意力矩阵（如最短路径距离偏置）。以 Graphormer 为例：初始嵌入 h_v^{(0)} = x_v + z^-_{deg^-(v)} + z^+_{deg^+(v)}（入度/出度的可学习嵌入向量），注意力系数加 bias_{φ(v,u)}（φ(v,u) 为顶点间最短路径距离的可学习共享标量）。这类编码使 GT 的注意力计算比文本 Transformer 更不规则（引入拓扑相关偏置），削弱块矩阵优化的有效性。
- TAGT 的关键设计：TAGT 对 SE/PE 语义无关（encoding-agnostic）——只要 SE/PE 在进入 TCU 前被物化为 per-vertex 稠密向量即可。FUU（Feature Update Unit）把原始特征与编码向量做同步取数+轻量拼接+线性投影生成统一基层嵌入；支持不同 SE/PE 方案只需输入维度适配与配置更新，无需微架构改动；甚至兼容需要在线图算子的编码方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- SE/PE 在 TAGT 中的 pipeline 位置：TDL 的 Fetch_Coding 6 级流水线从 HBM/片上 buffer 取结构编码向量 → FUU 与 Fetch_Features 取到的原始特征同步拼接 → 线性投影 → 基层（leaf）嵌入写入 TDS-CSR Table 并流式送 MOU 参与 fusion 顶点聚合。整个过程中编码只作为"辅助向量载荷"，不触发编码专用图算子。
- 计算示例（Graphormer）：h_v^{(0)} = x_v + z_{deg^-(v)}^{-} + z_{deg^+(v)}^{+}；注意力分数 s_{v,u} = h_v W_Q (h_u W_K)^T / √d_K + bias_{φ(v,u)}。TAGT 只需把这些 per-vertex 向量/偏置作为输入即可。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现：度嵌入（in/out-degree learnable embeddings）、最短路径距离（SPD）偏置、拉普拉斯特征等；本论文未给出新编码，直接复用 Graphormer 类方案并验证 TAGT 的编码无关性。
- 使用：作为 TAGT 前端输入（每个顶点一个稠密编码向量），无需改硬件；支持向量化 SE/PE 的不同维度仅需配置更新。评估中 4 个 GT 模型（GT/Graphormer/UGformer/EGformer）均以此方式接入。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
