## 五分钟规则与 break-even interval（数据放置经济学启发式）

术语解释
- 五分钟规则（Gray & Putzolu, 1987）是存储-内存经济学的数据放置启发式：当数据块的访问间隔低于 break-even 阈值 T_break-even 时驻留 DRAM 更经济，否则留在存储；T_break-even 由"DRAM 租金"与"反复从存储取数成本"的平衡点决定。论文从第一性原理重写该规则，把主机成本、可行性约束与工作负载纳入。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 经典形式：T_break-even = (每 MB 页数 ÷ 存储 IOPS) × (存储成本 ÷ 1MB DRAM 成本)。后续 revisit（1997/2007/2019）保持 economics-only 视角，把 DRAM↔SSD 阈值定在分钟级。论文的校准形式（Sec. III，Eq.1）：T_break-even = ( $CORE/IOPS_CORE + l_blk·$H_DRAM/B_H_DRAM + $SSD/IOPS_SSD ) · C_H_DRAM/(l_blk·$H_DRAM)，即"每 I/O 总成本（核 + DRAM 带宽 + SSD）÷ 每 MB DRAM 月租折算"，且 IOPS_SSD 与 $SSD 由第一性原理设备模型推导而非厂商数据表。核心发现：GPU+Storage-Next SSD 下阈值从分钟压到秒级（512B SLC 从 CPU+DDR ~34s 降到 GPU+GDDR ~5s）。
- 从系统架构角度拆解术语：该规则是"DRAM 缓存该缓存什么"的量化决策依据，输出单个阈值 T_break-even，系统据此把访问间隔更短的块放 DRAM、更长的放 flash。经典规则只算设备价格；论文把它扩展为系统级决策：把宿主 I/O 成本（核、DRAM 带宽）显式计入，再把可行性（主机 IOPS 上限、尾延迟 SLO）与工作负载访问间隔分布（RQ3）纳入，使其从"启发式"变成可操作的配置/升级指导。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：作为数据放置策略的成本模型——给定平台（核/DRAM/SSD 成本与时序）与工作负载（访问间隔、读写比、SLO），算出 T_break-even；配合 RQ3 的 T_B/T_S/T_C 三阈值判断平台是否可行/最优、该升级哪项资源（加 DRAM 带宽/SSD 吞吐/主机 IOPS/容量）。论文用归一化制造成本（DDR=1、GDDR=2、控制器 15、CPU 核 4、GPU SM 3）避免市场价偏差。信息缺口：论文未给出实际部署的缓存替换策略（LRU 等）细节——规则只提供阈值。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
