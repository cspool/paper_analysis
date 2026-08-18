## 两级级联 ECC（BCH 内码 + LDPC 外码，SSD 小粒度纠错）

术语解释
- 传统 SSD 用 4KB 码字保护数据（4KB codeword ECC），导致 ≤4KB 随机小读每次都触发整 4KB 页解码与搬运，IOPS 被压平；两级级联码改为"每 512B 扇区一个 BCH 内码 + 跨 8 个扇区的 LDPC 外码"，小读只解码所需 BCH 字并跳过 LDPC，消除 SSD 内读放大。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ECC 是 NAND 可靠性的基石（NAND 原始误码率随制程与层数上升，需 ECC 纠错）。论文把 ECC 组织方式与访问粒度强耦合：如果 ECC 码字=4KB 物理页，那么一次 512B 逻辑读也要把整 4KB 页搬出并全页解码，512B 请求的实际内部读放大为 8 倍，这正是"传统 SSD 在 ≤4KB 随机 IOPS 平直"的根因之一。MQSim-Next 采用两级级联码：BCH 内码按 512B 扇区，LDPC 外码跨 8 扇区；只触碰部分扇区的读解码所需 BCH 字即可，LDPC 只在 BCH 失败时升级触发（整 4KB 解码，增加传输与迭代解码延迟）。可调 BCH 错误概率参数 p_BCH 用于探索小读尾延迟与 ECC 诱导读放大。
- 从芯片设计角度拆解术语：ECC 解码引擎位于 SSD 控制器数据路径上，其处理粒度决定设备级随机 IOPS 的上限。级联码把"小请求小纠错、大错误升级大纠错"分层，使 512B 随机读的每请求解码成本与请求大小成正比而非与页大小绑定。模拟结果（Fig. 7d）显示 512B BCH 失败触发 4KB LDPC 解码时吞吐只适度下降，<1% 失败率下接近无错误平台。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- BCH 与 LDPC 均为成熟纠错码：BCH 为代数分组码（适合短码字、硬判决、低延迟），LDPC 为迭代软判决码（适合长码字、高纠错能力、较高延迟），两级级联（concatenated code）在 SSD 控制器中常见。论文在 MQSim-Next 中把它做成显式可配置模型（内码扇区大小、外码跨度、失败升级路径、p_BCH 参数），用于在 Storage-Next 小粒度 regime 下评估 ECC 对 IOPS 与尾延迟的影响；解析模型则在 IOPS 推导中假设 ECC 不构成瓶颈（xlat/PCIe/ECC 均配置为非限制项，瓶颈落在 NAND/channel 物理）。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
