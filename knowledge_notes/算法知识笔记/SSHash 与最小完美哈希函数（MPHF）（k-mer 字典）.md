## SSHash 与最小完美哈希函数（MPHF）（k-mer 字典）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 最小完美哈希函数（MPHF）是双射哈希：把 n 个键（这里为 n 个不同 k-mer）无碰撞地映射到 [0,n)，无空桶，空间接近理论下界（~1.44 bits/key，PTHash 等实现）。SSHash（Sparse and Skew Hashing of K-Mers，Pibiri 2022，Bioinformatics i185–i194）是基于 MPHF 的"压缩、关联、精确、带权"k-mer 字典：利用 k-mer minimizer 的稀疏与偏斜分布 + 最小完美哈希 + 紧凑编码，得到比此前序列字典明显更好的空间-时间折中，存储 unitig 为连续字符串并支持精确成员查询（每个字符串关联 [0,n) 唯一整数 ID）。GRAINS 采用 SSHash 作为 DBG 骨干字典（与 Fulgor 相同选择），并强调其技术依赖一般 DBG 性质、也可用于其他 k-mer 字典骨干。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- GRAINS 的查询利用 SSHash 的 Sizes 数组远小于 Offsets/Strings 的特性（Sizes 仅占大图 <4% 空间）：主机侧先查 Sizes 拿到 Offsets 索引、排序分批后再发 SSD，把对 Offsets 的访问变成顺序流——这是 Genome-Graph-Aware Query Reordering 的物理前提。查询流程伪代码：`h = mphf(minimizer(g))` → `lo = Sizes[h], hi = Sizes[h+1]` → 对 `i in [lo, hi)` 读 `Offsets[i]` → 在 `Strings` 的 `Offsets[i]` 处窗口 `k−m+1` 内比对 minimizer 与 k-mer → 命中则 unitig ID 即为该 k-mer 的关联整数 ID，取其颜色。注意：虽然整图太大无法进 host（§3 动机分析），Sizes 只占小部分，可在 host 安全驻留处理。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MPHF 由 PTHash 库（jermp/pthash，C++）构建；SSHash 用 2-bit 核苷酸编码、CMake 构建（`git clone --recursive https://github.com/jermp/sshash.git`，`-DSSHASH_USE_MAX_KMER_LENGTH_63=On` 支持 k≤63，`-DSSHASH_USE_ARCH_NATIVE` 提性能）。使用：构建对 k-mer 集的 MPHF 与 Strings/Offsets/Sizes 索引 → 查询时按 minimizer→h→Sizes→Offsets→Strings 定位。GRAINS 把 SSHash 结构存进 SSD（Strings/Offsets 低复用大数据放 NAND、Sizes 可放 host），并为其设计存储友好布局与调度。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
