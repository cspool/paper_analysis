## Minimap2（seed-chain-extend 长读基因组比对工具）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Minimap2（Li, H.，Bioinformatics 34(18):3094-3100, 2018，GitHub: lh3/minimap2）是通用核酸序列两两比对工具，按 **seed（播种）→ chain（链化）→ extend（扩展）** 三阶段流程工作，是长读（PacBio/ONT）比对与 de novo 组装的事实标准（NextDenovo、Canu、MetaFlye、Shasta 等内部用它做 all-to-all 比对）。三阶段：(1) Seeding——把 reference/query 切成滑动窗口、每窗口取字典序最小 k-mer（minimizer）作种子，匹配 minimizer 对（anchor）经内存哈希表随机查询发现；(2) Chaining——对 anchor 列表做 1D 动态规划：每 anchor 回看至多 N 个前驱、以 max reduction 计算 chaining score（奖励重叠、惩罚 gap），连出长链；(3) Extend——对每 chain 做 banded Smith-Waterman-Gotoh（affine gap，默认 20 kbp band）+ 逐单元 traceback，输出 SAM。de novo 组装中 Minimap2 承担 ~76% 运行时间与数量级更高的内存（NextDenovo 人类基因组实测）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 reference-based 比对流程：`minimap2 -ax map-ont ref.fa reads.fq > out.sam`（参考比对）或 `minimap2 -x ava-ont reads.fq reads.fq > overlaps.paf`（all-to-all，供 de novo 组装）。内部：① 索引——对 reference 所有 minimizer 建哈希表（可选 HPC minimizer 压缩同聚物跑）；② seeding——对 query 每窗口取 minimizer、查表得 anchor（高频 minimizer 可过滤）；③ chaining——1D DP 递推 chaining score、按 0.8 阈值保留主/次比对；④ extend——双向扩展 + ksw2 SIMD 加速的 banded SWG 精化比对。Lembas（ISCA'26）将其作为算法语义"金标准"：不引入任何 trade accuracy 的近似/启发式过滤，结果必须包含 Minimap2 相同配置下的全部结果；三阶段分别被外部内存 columnsort 播种加速器、流式 chaining 加速器（复用 Guo et al. FCCM'19）、tiled SWG 扩展加速器替换。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C 语言单文件可执行程序，SIMD（SSE/AVX/AVX-512）向量化 DP；`-k`/`-w` 控 minimizer 长度与窗口、`-A/-B/-O/-E` 控替换/缺口罚分、`-z` 控 chain 拆分阈值；内存 chunking 参数可调（降低内存但跨 chunk 不做匹配检查、输出质量下降——Lembas 论文实测其 7× 内存与质量差异）。使用场景：reference-based read mapping、all-to-all 重叠检测（de novo 组装）、全基因组比对（WGA）；是 BWA-MEM 之外长读领域的对标基准。后续改进见 Li 2021（Bioinformatics 37:4572，"New strategies to improve minimap2 alignment accuracy"）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
