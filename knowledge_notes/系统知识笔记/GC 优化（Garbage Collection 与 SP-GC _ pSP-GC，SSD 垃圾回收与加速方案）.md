## GC 优化（Garbage Collection 与 SP-GC / pSP-GC，SSD 垃圾回收与加速方案）

术语解释
- GC 是 SSD 因 NAND 不能原地覆写而必须的后台机制：把 victim block 中仍有效的页迁移到新块后擦除旧块，腾出可写空间；GC 期间迁移操作阻塞 host I/O。SP-GC 用 SLC 模式加速有效页迁移但损失容量，pSP-GC（LOONG）用 pSLC + 长跨度重编程在保持容量下加速迁移。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NAND 必须先擦除（块粒度）才能重写（out-of-place update），因此 SSD 后台需要 GC：选定 victim block，把其中有效页读出迁移到空闲块，再擦除 victim block 回收空间。GC 的迁移写入量取决于块内有效页比例，有效页越多迁移成本越高；迁移用 OSP（1100 µs/页）写 TLC 时，GC 阻塞时间（host I/O 被阻断）很长，是随机/写密集负载下 SSD 性能的主要瓶颈之一。SP-GC（SLC-Program GC，[38]）：用 SLC 模式（96 µs/页）快速迁移有效页，但 SLC 块只保留 1/3 TLC 容量，剩余 2/3 容量无法使用，需额外补偿 GC 回收空间（论文实验：GC 数增 1.25 倍、有效页迁移增 1.48 倍，抵消了单次迁移提速）。pSP-GC（LOONG 的 GC 优化）：有效页以 pSLC 模式（114 µs/页，SLC 级速度）迁入空闲块，随后用长跨度重编程（955 µs/页）把 pSLC 块恢复为 TLC 全容量——不损失容量、无需补偿 GC。触发约束（时间/空间双域）：时间上仅前台 GC 时触发 pSP-GC（最小化 host 阻塞）；空间上每次 pSP-GC 只用 1 个块存 pSLC 有效页（实验：>71.5% 的块有效页 < 1/3，单块足够；有效页超 1/3 时用冷 host 写或 bg-GC 页参与重编程）。
- 从系统架构角度拆解术语：GC 是 FTL 管理的后台调度机制，其优化目标是"减少 GC 触发次数 + 降低每次 GC 的迁移成本 + 消除迁移带来的容量损失"。方案对比（论文 Fig. 12/13，事件驱动模拟器 + MSRC/MSPS/FIU/YCSB）：baseline（OSP 同 plane 顺序迁移）、SP-GC、HS（热度分离）、8-stride（重编程 + SP-GC、stride 限 8 WL，带/不带 HS）、pSP-GC（带/不带 HS）。结果：SP-GC 因补偿 GC 反而比 baseline 慢 1.05×；HS 降 25.0%；8-stride w/o HS 降 8%、pSP-GC w/o HS 降 24.0%；叠加 HS 后 pSP-GC 降 20.7%（vs HS）、总计较 baseline 降 37.5%。pSP-GC 使 51.9% 的有效页迁移受益于 pSLC（8-stride 仅 26%），GC 数较 baseline 降约 20%，99 百分位读/写尾延迟降 20%/18%，IOPS 提升 1.8×（最高 2.9×）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：GC 逻辑在 FTL/控制器固件中（victim 选择、有效页枚举、迁移调度、块擦除）；pSP-GC 在 FTL 中增加 RBP/SP/RP 指针与 pSLC/重编程调度（详见 FTL 条目 LOONG 补充）。使用边界：GC 不敏感场景（小负载、低更新率如 CFS/SRC1_2/homes，或降低 warm-up 比例的合成负载）收益有限（3–3.6%），但可转由编程优化获益；无冷数据最坏场景下 pSP-GC 退化为 HS 性能（仍较 baseline 降 21.8%），不会更差。

涉及论文标题：
- LOONG: Utilizing Long-Stride Reprogramming to Enhance the Performance of SSDs
