## 主动数据层（Active Data Tier，内存层次结构重构）

术语解释
- 主动数据层指把 NAND flash 从"被动容量层"升格为内存层次的"主动扩展层"：当 DRAM↔flash 缓存阈值从分钟压到秒级后，flash 能承担内存级数据路径（高随机 IOPS、低延迟、高容量），算法与数据结构的访问路径、放置与调度需围绕"秒级复用"重新设计。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 传统层次观：DRAM=内存（低延迟、高成本）、SSD=存储（高延迟、低成本），"tape is dead, disk is tape, flash is disk"；缓存阈值在分钟级意味着只有低频数据值得驻留 DRAM，flash 只做批量传输。论文核心论点：GPU 主机（高 IOPS 预算）+ Storage-Next SSD（512B 高 IOPS）+ 成熟 NAND 物理使缓存阈值降到几秒，flash 的 IOPS/\$ 使其能服务"内存级"随机访问，从而成为主动层。两个设计原则：(1) 利用超高小块随机 IOPS，偏好细粒度访问与并发；(2) 利用 flash 远低于 DRAM 的 $/GB，允许稀疏/过度供给结构换取速度与简单。案例：SSD 原生 KV（无 DRAM 索引）与两阶段渐进 ANN（reduced/full 双档向量）。
- 从系统架构角度拆解术语：这是内存层次管理与数据放置的系统级范式转变——"缓存阈值秒级"意味着数据放置决策频率更高、flash 参与访问路径更深（不再只做整块批量读写），系统软件（KV、向量检索、embedding 缓存、长上下文元数据）需重设计访问路径/数据结构/调度以利用秒级复用。论文 RQ4 列出开放设计空间：高 IOPS 下的访问路径与调度、层级感知的数据布局与放置、针对秒级复用的轻量排序/一致性/恢复、多租户 QoS/公平/隔离。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：依赖硬件侧（Storage-Next SSD、SCADA GPU 直连、HBF 等）+ 软件侧（零拷贝 I/O、SSD 原生数据结构、GPU 发起存储访问）协同。论文用两个案例示范：KV 达内存级吞吐（100+ Mops/s，FASTER 水平）、ANN 达几十 KQPS（DiskANN ~5 KQPS 量级）。论文为框架+案例的示范性工作，未给出可部署系统；未来方向包括耐久性/能耗 TCO、多租户、CXL/NVMe-oF 织物扩展。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
