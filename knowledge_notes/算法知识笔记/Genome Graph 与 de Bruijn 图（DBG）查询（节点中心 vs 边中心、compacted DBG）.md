## Genome Graph 与 de Bruijn 图（DBG）查询（节点中心 vs 边中心、compacted DBG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Genome graph（基因组图）用图 walk 而非独立线性序列表示基因组数据库：把数据库条目间共享的子序列折叠成单一节点、相邻子序列用边连接，节点元数据（metadata）标明哪些条目包含该序列，可把序列集压缩上千倍，表达力强（编码进化历史与多样性、降低 bias、提高分析准确率），并利用冗余避免对共享序列的重复计算。与一般图（社交/网页/路网，幂律度分布、需显式存边、可节点重排）不同：基因组图节点度最多 4（A/C/G/T 固定字母表决定）、节点或边隐式获得、且"查询本身是生物序列"带来共享子串 k-mer 映射到索引附近区域的结构耦合局部性。de Bruijn 图（DBG）是重叠图的一种，近年大规模基因组图分析的主流：节点=唯一的 k-mer（长度 k 子串），节点 u→v 有向边当且仅当 u 的 (k-1) 后缀等于 v 的 (k-1) 前缀；compact DBG 把最大非分支路径（unitig）合并成单一节点以缩小规模。两种表示：节点中心（node-centric，如 Fulgor：存全部 k-mer 节点、边隐式定义）与边中心（edge-centric，如 MetaGraph：存 (k-1)-mer 节点、只存表示观察到的 k-mer 的边）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 查询 pipeline（k-mer set lookup，GRAINS 图 7 流程）：给定 read 提取 k-mer → 对 k-mer g 计算 minimizer（哈希最小的 m-mer）→ 用 minimizer 的最小完美哈希值 h 索引 Sizes → 由 Sizes[h+1]−Sizes[h] 得 Offsets 区间 → 读 Offsets 得 Strings 中页内偏移 → 在 Strings 的 k−m+1 窗口内找到 minimizer 并校验其余 k-mer → 命中则取 unitig 的颜色（元数据）。read mapping 分 alignment-free（全部 k-mer 匹配→汇总元数据→分类 read，大样本研究常用）与 alignment-based（k-mer 命中定位候选区域后再做近似字符串匹配/动态规划精化，可集成 SeGraM 对齐加速器）。GRAINS 的 DBG 数据布局：unitig 按预定顺序存连续字符串（Strings），Offsets/Sizes 辅助定位，颜色按 unitig 排序后用 Color Bitmap 标记每色起点。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 工具：Fulgor（节点中心，SSHash 字典，github.com/jermp/fulgor）、MetaGraph 框架（边中心，github.com/metagraph-rs/metagraph）、GRAINS（用 SSHash 做字典）。图构建自 MetaSUB Consortium 全球样本子集（G_MetaSUB，含颜色 Fulgor 659 GB / MetaGraph 822 GB）与 SRA 公共代表子集（G_SRArep，161/231 GB）。用途：物种/病原体鉴定（k-mer 集合查找）、个性化医疗、宏基因组与废水监测、群体规模病原体监测。DBG 是无损编码，各工具（Fulgor/MetaGraph/GRAINS/IdealAccMem）查询精度一致；变体图工具（VG、minigraph）在遗传多样性大时不缩放，不在大数据库场景使用。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
