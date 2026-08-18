## Single-Device-Per-Channel（SDPC，单设备每通道）与 32B 粒度冗余预算

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SDPC（single-device-per-channel）是 HBM、LPDDR 等内存的通道组织方式：一条内存通道只挂一个 DRAM 设备（die/堆栈），以换取更高带宽与更低功耗。逻辑链与代价：DDR 模块把 64B 数据跨多个 device 分布，单设备失效可由 rank 级保护（SDDC/chipkill）扛住；SDPC 把整条通道的数据都放在一个设备里（"all eggs in one basket"），任何设备级缺陷都会毁掉整条通道，设备级 SDDC 不可行，S-ECC 只能瞄准大粒度突发或链路级故障。同时数据密集型负载把最小取数粒度压到 32B（vs DDR 的 64B）以抑制 overfetch、提高有效带宽，但更小的粒度意味着 ECC 开销的摊销空间变小：SDPC 内存通常每 32B 块只有 2B 冗余（6.25%）预算，而 DDR 常见每 64B 8B（12.5%）。
从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
SDPC 使 O-ECC 与 S-ECC 的粒度天然对齐（都在 32B 块上），这是 HBM/LPDDR 区别于 DDR 的关键结构特征，也决定了三层 ECC 的冗余配置方式。本文给出的两个实例配置（每 pseudo-channel、每 32B 数据块）：HBM4 = 2B S-ECC + 4B O-ECC + 1B L-ECC（合计 18.8% 存储开销、15.6% 传输开销）——冗余大头给 O-ECC 做 16-bit 符号纠错，S-ECC 只剩 2B（SEC-DED 或 CRC16）；LPDDR6 = 2B S-ECC + 2B O-ECC + 2B L-ECC（12.5%）——O-ECC 用 2B 对 32B 数据及其 S-ECC 冗余做 SEC-DED，外部每 burst 传 36B（12 I/O pin × 24 拍：32B 数据 + 2B S-ECC + 2B L-ECC/DBI）。Cerberus 的目标正是这个结构：在 SDPC 的 32B 单元上把总冗余压到 12.5%（32b 共享冗余，另评估 40b/15.6%），并证明比 HBM4 的 18.8% 分层配置更强。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SDPC 是芯片组织选择（HBM 堆栈/ LPDDR 封装按单设备通道供电与布线），不是可配置项；其上的冗余预算是架构设计参数。使用上的设计要点（本文归纳）：① 冗余在各层的分配决定系统上限——HBM4 把 32b 全给 O-ECC 后 S-ECC 只能 SEC-DED/CRC，out-of-bank 错误无解；② 32B 粒度下可用的码很短（288b 码字、32b 冗余），强纠错必须符号化（16-bit 符号对齐物理故障域，如 SWD 故障宽度 8–32 位）；③ 分层冗余不可复用是低效根源，Cerberus 用共享冗余打破这一限制。评估上的使用：GPU 模拟（本文用 Accel-Sim 配 V100 + 32 通道 HBM4）与能耗核算都要按 SDPC 的传输位宽差异建模（HBM4 内部传 256+32+16、外部传 256+16；Cerberus 内外均 256+32）。
涉及论文标题：
- Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection
