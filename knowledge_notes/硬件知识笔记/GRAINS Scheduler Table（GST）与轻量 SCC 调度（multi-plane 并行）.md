## GRAINS Scheduler Table（GST）与轻量 SCC 调度（multi-plane 并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GST 是 GRAINS 放在 SSD 内部 DRAM 的每 die 一个小调度表：每 Strings 页一行，行内容为——页内位地址（Offsets 条目标记）、k-mer minimizer 及前后缀（来自压缩 k-mer 表示）、full 标志（满则指向扩展表）、one-hot 编码的目标 plane 位图。调度原理：控制器上的轻量 ISP 调度逻辑逐行顺序读每个 GST、round-robin 把请求发到所有 die/plane，天然合并同页访问（避免冗余页访问）、最大化 die 级并行，且支持 multi-plane 操作（one-hot 位图让同一 die 的多个 plane 并发服务不同访问）——不需要 SSD 内的排序加速器（元素量大、内部 DRAM 带宽受限，排序加速器低效）。GST 能放进内部 DRAM 的两前提：GRAINS 块粒度 L2P（§8）释放大部分 DRAM + 压缩 k-mer 表示让行很小；10M reads 大查询仅需 2.9 GB < 4 GB 内部 DRAM（典型 4-TB SSD），超出时 host 保留 Sizes 批次直到 SSD 处理完当前批。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Strings 查询调度（图 11）：① 调度器读每 GST 行的 Strings 地址+对应 k-mer，把 k-mer 发到对应 die（2），并递增该行指针供下次访问；② die 读 Strings 页 → ECC_LITE（3）→ IFP 比较单元对页内窗口与传入 k-mer 做位级比较（4）；③ 比较结果+命中 unitig ID 回 SSD 控制器（5）。由于 unitig ID 按页序回到内部 DRAM，Colors 可顺序扫描 Color Bitmap 获取（§7.3）。round-robin 跨 16×8×4=512 平面分发 + multi-plane 并发 = 高内部带宽利用；同页访问被合并 → 每块最多读一次 → 还避免读干扰错误。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为内部 DRAM 表 + 控制器轻量 ISP 调度逻辑，属于"repurposing existing SSD structures"（复用 FTL 释放的 DRAM 空间）。消融验证：GRN-B-S（batching+scheduling，即 GRN-Ext）比 GRN-B（仅 batching）平均再快 2.3×，归因于对 Strings/Colors 的存储友好访问；GRN-B-S-SCC 完整版再 2.0×。调度只需简单顺序读表+round-robin，无复杂逻辑，符合 SSD 有限硬件资源约束。

涉及论文标题：
- GRAINS: Enabling High-Performance and Low-Cost Graph-Based Genome Analysis via Storage-Aware Algorithm-Architecture Co-Design
