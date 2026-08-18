## 写放大（Write Amplification，Φ_WA）

术语解释
- 写放大指因 NAND 必须先擦除后写、且以整页/整块为粒度，SSD 实际写入 NAND 的数据量（含垃圾回收导致的页迁移）超过主机请求写入数据量的倍数，记为 Φ_WA≥1。它直接抬高每单位有效写入的 NAND 磨损与通道/带宽占用。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NAND 不支持原地覆写：主机写请求先落在新物理页，旧页数据成为无效页，后台垃圾回收(GC)把含有效数据的块内页搬迁到新块并擦除旧块，这些搬迁写入构成背景写流量。3D NAND 更大的页/块（如 16KB 页、768 页/块）使 GC 效率变差、写放大上升（网络来源：页/块越大、每 GC 浪费页越多，写放大实测可在 1.7~19.9 之间变化）。论文把 Φ_WA 纳入第一性原理 IOPS 模型：读写混合下读占比 R_r=(Γ_RW+Φ_WA−1)/(Γ_RW+2Φ_WA−1)、写占比 R_w=Φ_WA/(Γ_RW+2Φ_WA−1)，单 die 峰值 IOPS = R_r·N_Plane/τ_sense + R_w·N_Plane·l_PG/(τ_prog·l_blk)。在 Γ_RW=90:10、Φ_WA=3 的保守假设下，SSD 有相当部分 NAND 能力用于写路径与 GC 流量而非服务主机读。
- 从芯片设计角度拆解术语：Φ_WA 把"主机可见 IOPS"与"NAND 物理 IOPS"解耦——NAND 物理能力是上限（sensing/program 延迟、平面数、通道数决定），主机可用的读/写 IOPS 需扣除 GC 迁移与写合并开销。MQSim-Next 模拟显示只读→90:10→70:30→50:50 时 IOPS 从 82M→68M→52M→34M，正是 GC 流量与主机 I/O 竞争的量化体现；解析模型取 Φ_WA=3 因而略低于模拟值。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现上，SSD 通过 FTL 的磨损均衡、GC 调度（如 idle GC、preemptive GC）、over-provisioning（预留块）与写合并降低 Φ_WA。论文在解析模型与 MQSim-Next 中都用 Φ_WA 作为参数：解析模型固定 Φ_WA=3（保守），模拟器显式建模 GC 与磨损均衡，两者对比验证模型假设。论文还指出未来方向：把耐久性（retention/refresh 策略、寿命折算）与每 I/O 能量纳入 TCO 建模，形成可持续性感知的配置工具。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
