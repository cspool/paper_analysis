## 两阶段渐进式 SSD-resident ANN 搜索

术语解释
- 两阶段渐进式（two-stage progressive）SSD-resident ANN：每个 embedding 在 SSD 上同时存 reduced-dimension（如 512B）与 full-dimension（如 2-8KB）两种形式，查询先取 reduced 向量粗筛淘汰大部分候选，再仅对少量 promoted 候选取 full 向量精排，把 IOPS-bound 的小块读与带宽-bound 的大块读分层。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 动机：SSD 驻留 ANN 中，多数距离比较只是"确认拒绝"（Gao et al. 报告 >90% 比较用于淘汰候选），全维度求值往往不必要；先降维淘汰、再全维精排可大幅减少大块读。reduced 向量来源：(1) 线性变换（PCA、随机投影）、(2) 双模型 embedding pipeline、(3) MRL（原生支持多分辨率向量）。论文在 MRL 生成的 MS MARCO、20 Newsgroups、DBpedia 语料上验证 recall>98%。Storage-Next 直接受益：绝大多数访问命中 512B reduced 向量（小块随机读 → 极高 IOPS），promoted 子集（5%-20%）带宽-bound 但被大拒绝率摊薄。
- 从算法pipeline角度拆解术语：一次查询的 pipeline：①取查询的 reduced 向量与候选集的 reduced 向量计算距离 → 淘汰 >90% 候选（小块随机读、IOPS-bound）→ ②对 promoted 子集取 full 向量重排（大块读、带宽-bound）→ 返回 top-k。论文的量化评估（Fig. 10）：512B→2KB（95%/5%）、512B→4KB（90%/10%）时 GPU+SN 保持 SSD-IOPS 受限（7-11→13-17 KQPS 随 DRAM 512GB）；512B→6KB（85%/15%）400GB 后 GDDR 带宽封顶（8.3 KQPS）；512B→8KB（80%/20%）300GB 即带宽受限。promotion 率越高、DRAM 流量越大、带宽天花板越早出现。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：embedding 双份存储（reduced+full）、查询端两阶段执行、promotion 判定（距离阈值或 top-k 动态）；reduced 版本可由 MRL 直接截断得到（无需额外训练）。论文用途：展示"flash 作为主动层"如何催生新算法——低成本的 TB/PB 级 embedding 表留驻 flash，GPU+SN 把吞吐推到几十 KQPS（DiskANN ~5 KQPS 量级）且保持 HNSW 级 recall。论文为模型驱动评估（分析框架+MQSim-Next 模拟），无开源实现。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
