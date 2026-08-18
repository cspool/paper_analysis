## k-mer 与 Minimizer（基因组序列查询）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- k-mer 是 DNA/RNA 序列中长度为 k 的连续子串（k 常见 21–63），测序产生的 read 被切成全部 k-mer 作为与大规模数据库比对/查找的基本单元。Minimizer 是给定窗口内哈希值最小的 m-mer（m<k）：对每个 k-mer 取 minimizer 可得到位置确定的代表性子串，使"共享 minimizer 的 k-mer 在字典/索引中聚在一起"，作为路由键大幅减少索引比较量。k-mer minimizer 具有稀疏且偏斜分布的统计特性，是 SSHash 等紧凑 k-mer 字典（空间-时间折中）的设计基础。GRAINS 利用该特性做主机侧查询重排：按 Sizes 排序后连续 k-mer 共享同一 minimizer，传输时只存一次 minimizer、其余只留差分，将 host→SSD 传输数据量平均压缩 2.3×。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 查询/压缩流程（GRAINS 图 7/9）：① 对每个 read 提取全部 k-mer（2-bit 编码，每 k-mer 一个整数）；② 对 k-mer g 求 minimizer：在 g 的 k−m+1 个 m-mer 中取哈希值最小者（如 m=15 于 k=31）；③ 用 minimizer 的最小完美哈希 h 索引 Sizes[h]；④ 排序/分批后，批次内连续 k-mer 共享 minimizer 时只传 minimizer+差分（例：同一 minimizer 下 5 个 k-mer 只传 1 个 minimizer + 4 个差分子串）；⑤ die 内 IFP comparison 把 k-mer 与 Strings 窗口做位级逐位匹配。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：k-mer 提取把 read 每个长度 k 窗口编码为 2-bit 整数；minimizer 用滚动哈希或最小哈希（如 MinHash/ntHash 类）对窗口内 m-mer 取最小。工具：SSHash（jermp/sshash，C++，2-bit 编码、默认 k≤31 可编译到 63）、Fulgor（Seeding 用 minimizer，类似 MinSeed）。使用场景：k-mer 集合查找、read mapping seeding、序列去重、宏基因组分类。GRAINS 强调 k-mer 提取在 host 完成（SSD 内频繁写会缩短 NAND 寿命），压缩/排序也是 host 软件步骤。
- Lembas 补充视角（ISCA'26，Minimap2 播种阶段）：Minimap2 在滑动窗口内取**字典序最小 k-mer 作为 minimizer 种子**，用**内存哈希表**做随机查找发现匹配 anchor——哈希访问不可预测随机、表必须整体驻留内存，是 seed 阶段内存容量瓶颈（Minimap2 靠 memory chunking 限内存但跨 chunk 不做匹配检查、降低输出质量）。Lembas 的播种加速器**完全移除哈希表**：minimizer parse 产出 16 B 〈minimizer, index〉 元组流，经 PCIe 溢出到 NVMe，用外部内存 columnsort 按 minimizer 字典序全局排序，reference/query 两有序流做**流式 zip 匹配**（顺序扫描）得 anchors → 内存需求恒定 ~8 GB（7× 降低）。代价：刻意不做 Minimap2 的启发式 anchor 过滤（需随机访存），下游工作量放大（人类基因组 7.06× 更多 chains）。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
