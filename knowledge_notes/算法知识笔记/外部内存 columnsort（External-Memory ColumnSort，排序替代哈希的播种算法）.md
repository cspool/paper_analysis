## 外部内存 columnsort（External-Memory ColumnSort，排序替代哈希的播种算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Columnsort（Leighton, "Tight bounds on the complexity of parallel sorting"，1984/85）是面向"高瘦" r×c 网格的并行排序算法：把全局排序拆成反复"列内排序 + 网格转置/移位"——列内排序各列互不通信、天然可并行；转置后同一批值落入不同列，多轮后全局收敛。约束 $r \ge 2c^2$（网格必须高瘦）。经典 8 步：排序列 → 转置 → 排序列 → 逆转置 → 排序列 → 前移 ⌊r/2⌋ → 排序列 → 回移。**外部内存变体**（Lembas 新颖点，ISCA'26）：数据集超出 FPGA 片上/HBM 容量时把 256 MB 列存于 NVMe、经 PCIe 来回搬运（排序 + 转置），设计取舍随之外移（用简单 16-to-1 单发射 merger 即可达 PCIe 上限，无需复杂宽发射 merger）。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Lembas 用它把 Minimap2 播种的"内存哈希表随机查找"替换为"全量外部排序 + 流式 zip"：① minimizer parse 产出 16 B 〈minimizer, index〉 元组流 → 溢出存 NVMe；② columnsort 加速器：数据组织成 r×c 网格（$r \ge 2c^2$，256 MB HBM bank 上限 → 可排序 ≤512 GB，超出按 512 GB 分块后软件合并），16 个 16-to-1 单发射 merge-sort kernel 各独占一对 HBM pseudo-channel（1 数据 + 1 scratchpad），250 MHz/4 GB/s 每 kernel，6 次 sweep 排序 256 MB 列；③ 4 轮列排序 + 3 次转置回传（host 侧多 KB 大块 memcpy 重组转置列）→ 8 GB/s 双工 PCIe 上限 → 有效端到端排序吞吐 ~2 GB/s；④ 两有序 minimizer 流流式 zip 匹配得 anchors → 按 idxR 二次 columnsort（供 chaining 用）。16-to-1 是带宽/芯片面积最优：更大 fan-in 减 sweep 但装不下、更小 fan-in 需更多 pass；16 kernel 恰好饱和 U50 的 ~8 GB/s 双工 PCIe。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 一般实现：FPGA 上既有 Hypersort（ICFPT'22，HBM-FPGA 内存内实现）；CPU/分布式系统上用 out-of-core 变体（Dartmouth "Stupid Columnsort Tricks"：4-pass、s 不必整除 r、$r\ge 4s^{3/2}$ 可松弛）。Lembas 的用法：seeding 加速器（minimizer 解析与 anchor 匹配简单、与 columnsort 共享同一 bitfile），资源占用 Seed 361,624 LUT (41.53%)/517 BRAM (38.47%)（表 IV）。效果：seed 内存恒定 ~8 GB（7× 降低、无 Minimap2 chunking 质量损失）、seed 性能比 mm64 快 70%（比 G³SA 慢 15%）。使用场景：任何"哈希随机访问成为内存容量瓶颈、可转化为排序+流式扫描"的数据密集型比对/去重。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
