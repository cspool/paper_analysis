## Cross-Read K-Mer Batching 与 Genome-Graph-Aware Query Reordering（GRAINS 主机侧查询优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GRAINS 的两项主机侧算法优化：(1) Cross-Read K-Mer Batching——不再逐条 read 查询，而是把同一 read set 中不同 read 的 k-mer 合并批量查询，用轻量数据结构维护 k-mer→所属 read 的映射，图查询返回匹配 k-mer 的颜色后按 read 汇总，从而成量减少对图节点的随机访问次数（利用"不同 query read 的共享子串 k-mer 映射到索引附近区域"的基因组图特性）；(2) Genome-Graph-Aware Query Reordering——利用最小完美哈希 k-mer 字典（SSHash）中 Sizes 数组远小于 Offsets/Strings 的特性，先在 host 用 Sizes 完成 k-mer 查找、拿到 Offsets 索引，据此把 k-mer 排序并切成等长 disjoint 批次，使 SSD 侧对 Offsets 的访问变成顺序流；排序时利用"按 Sizes 排序后连续 k-mer 共享同一 minimizer"只存一次 minimizer+差分，把 host→SSD 传输量平均压缩 2.3×。二者与数据传输/查询构成流水线：一批排序与上一批传输和 Offsets 查询重叠。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 执行流程（图 9）：① host 提取每 read 的 k-mer；② host 查 Sizes（用 minimizer 的 MPHF h）得到 Offsets 索引；③ 按 Sizes 值把 k-mer 切成等长 disjoint 批次、排序（该批次排序与上一批传输/查询流水线重叠）；④ 批次内同 minimizer 只传一次+差分压缩；⑤ 批次经标准 NVMe 数据路径送 SSD 内部 DRAM（不写 flash）；⑥ SSD 顺序访问 Offsets、经 GST 调度查 Strings、ISP 扫 Color Bitmap；⑦ 结果回 host，按 k-mer→read 映射把颜色汇总到每个 read 完成分类。k-mer 提取与排序在 host 完成（SSD 内频繁写会缩短 NAND 寿命）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为 GRAINS 查询准备阶段的 host 软件（AMD EPYC 7742 + 1.5 TB DRAM 实测），排序/压缩开销由流水线隐藏（论文称需保证排序与传输不引入显著开销）。消融验证：GRN-B（仅 batching）平均 1.5×/2.2× 超 FG/MG（改善 Offsets 访问）；GRN-B-S（batching+scheduling）再 2.3×（Strings/Colors 存储友好）；GRN-B-S-SCC 完整版再 2.0×。GRN-Ext（优化在 SSD 外、PCIe 16 GB/s）也因存储友好执行流获得 3.4×/5.0× 平均加速，证明优化本身（不依赖 ISP/IFP）即有价值。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
