## 硬件内存压缩（Hardware Memory Compression）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 硬件内存压缩指内存控制器在 DRAM 存储前透明地对数据做无损压缩，从而在物理内存不变的情况下逻辑扩大内存容量（本论文：DRAM 占数据中心基础设施成本 40–50%，压缩是缓解 DRAM 成本/密度瓶颈的主流方案之一）。内存控制器将压缩后的数据打包进 DRAM，对软件透明；因为解压位于缓存缺失关键路径上（每个从压缩页取回的 64B 块都要等解压完成），解压延迟直接影响应用性能。本论文定义 DRAM block = 机器物理层的 64B 对齐块；压缩后原 64B 对齐块变成变长、变对齐，需要额外地址翻译层（压缩翻译条目 CTE）把 OS 物理页映射到压缩 DRAM 中的机器物理位置。
- 从硬件架构角度拆解术语：硬件压缩层位于 LLC 与 DRAM 之间的内存控制器。读路径：cache miss → 地址翻译（查 CTE 定位压缩页）→ 从 DRAM 取压缩数据 → 解压 → 返回 LLC。写/迁移路径：页压缩后经 zsmalloc 式分配器（变长压缩页打包，放置开销 1–2%）写入 DRAM。为控制解压成本，多个系统把 DRAM 组织成分层（tiered）：压缩页被访问时解压并提升到未压缩层（expand-on-access），后续访问以正常 DRAM 延迟命中。压缩率 c 的容量增益：每物理字节可存 c−1 额外字节逻辑数据。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：页级序列压缩（LZ/Deflate，行业规范强制）配合符号压缩；块级算法 CPACK/BDI/BPC 压缩率低。系统侧：TMCC（Deflate ASIC，MICRO'22）、DyLeCT（ISCA'24，压缩冷页/热页不压缩+expand-on-access）、Compresso/LCP 等；Linux zswap + 压缩卸载引擎是 OS 管理方案（有页错误开销，见系统架构层条目）。本论文 RST 把 per-page 字典压到 128B 使解压延迟从 140ns（ASIC Deflate 半页）降到 18ns/块，全系统仿真（gem5+Ramulator，DyLeCT 基线）平均性能 +15%、压缩数据访问延迟 ~110ns vs baseline 260ns~1µs。
涉及论文标题：
- Random-Access Hardware Sequence Compression
