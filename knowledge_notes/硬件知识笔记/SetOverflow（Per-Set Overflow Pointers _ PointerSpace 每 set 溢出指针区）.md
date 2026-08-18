## SetOverflow（Per-Set Overflow Pointers / PointerSpace 每 set 溢出指针区）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SetOverflow 是 Dorado 为"少量多共享者行"设计的目录项指针扩展：每个组相联目录 set 配一个 PointerSpace，含两个数组——SharerPointer（溢出指针池）与 OwnerWay（所有权表，T1 项，每项使对应 way 获得 T2 个连续 SharerPointer 条目）。某 way 的内建指针（论文为 2 个 6b 指针）耗尽时置 O（Overflow）位并 claim 一个 OwnerWay 条目（写入自己的 way ID）；可继续 claim 更多；PointerSpace 全满时置 B（Broadcast）位、清 O 位并释放自己的溢出指针，退回广播失效。Dorado 参数：T1=6、T2=2，每 set 12 个 6b 溢出指针；写事务且 O=1 时目录访问额外 +3 cycles。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程：O=0（M/E 态、S 态少共享者、或 B 位已置）不访问 PointerSpace；读 S 态 O=1 的行时先发数据、再访问 PointerSpace（不在供数关键路径）；写 S 态 O=1 的行时在关键路径——并行比较全部 T1 个 OwnerWay 项与目标 way ID，命中项读出对应 T2 个 SharerPointer，累加成数组后发无效化，后台清对应项与 O 位。对比多共享者方案（论文 Fig.11）：SCD（root/leaf 层级条目、set-associative 下冲突多，+3.1%）、Way Combining（须整体借用同 set 未用行的全部指针、cache 高占用时无未用行可用，+5.8%）、Pool directory（全局池需仲裁、条目须连续分配、需压缩/迁移逻辑、条目格式多样）——SetOverflow 每 set 本地、细粒度、无连续性要求、统一 6b 指针格式，+10.5%。可扩展性：32cl_32co 下仅 2% 数据 set 使 PointerSpace 溢出（16cl_64co 为 3%），扩簇不需扩 PointerSpace。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（论文）：SST 周期级建模 + TLA+ 验证（含 overflow pointer 分配/回收与"way 状态与 PointerSpace 一致"性质）；目录总量约 4.5% LLC 大小（270KB/核）。使用要点：O 位/B 位形成"精确跟踪→细粒度溢出→广播兜底"三级降级；写路径 +3 cycles 是可接受的代价（读不阻塞）；可作为对全位向量 UpperBound（66b/项）的低存储替代——Dorado 24b/项即达到其 99% 性能（目录存储少 2.75×）。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
