## 目录共享者跟踪表示：全位向量 / 有限指针（Dir_nB）/ 粗粒度向量

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
目录项如何记录"哪些 cache 有该行"决定目录存储与流量：全位向量（full-map）每核 1 bit 精确跟踪，N 核需 N bits/项，存储随核数线性、随 cache 总量二次方增长；有限指针（limited pointer，Dir_n）每项只存 n 个核 ID 指针；Dir_nB 在指针耗尽时置 Broadcast(B) 位，此后写需广播无效化到所有核（流量大、clean 驱逐不能静默）；Dir_nNB（no-broadcast）溢出时强制无效化一个共享者腾出指针（读也会引发无效化）；粗粒度向量每 bit 代表一组核（如 64 核/bit，SGI Origin 2000）。Web 证据（Simoni & Horowitz, ISCA 1991, https://ieeexplore.ieee.org/document/1021623）：大规模机中靠广播的 Dir_iB 即使溢出很少也表现差；Gupta 等在 17b/项预算下比较了 Dir_3CV_2/Dir_3B/Dir_3NB/Dir_32 的无效化分布。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dorado 的运用：baseline Dir2B 即 Dir_2B（2×10b 指针/项，23b/项）——共享者超 2 即广播失效，是 1.36× speedup 的对照；Dorado 用簇化把指针缩为 5b（核 ID）或存簇 ID（LRptr 一个指针覆盖 32 核），同等存储换来更多指针 + 聚合的远端共享者，使无效化消息 -39%；SetOverflow 在 24b/项内提供"2 内建指针 + 每 set 12 溢出指针"的等效 3 指针/way，逼近全位向量。上界对照 UpperBound：63b 全位向量（32 本地核 + 31 远端簇）+ 开销 = 66b/项，性能仅比 Dorado 高 ≤1%，但目录存储多 2.75×。取舍链：存储效率（有限指针胜）↔流量（广播/强制无效化劣）↔精确度（全位向量胜但不可扩展）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：商用目录普遍用有限指针/稀疏目录 + 溢出手段（Intel 在 DRAM 中放 directory bits 过滤远端 snoop）；多共享者扩展方案见 SCD（root/leaf）、Way Combining、SpongeDirectory、Pool directory 与 SetOverflow。使用要点：按"大多数行少共享者、极少数行多共享者"的分布（论文引 [22]）配置内建指针数与溢出容量；指针位宽 = log2(核数/簇数)，簇化可成倍降低位宽。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
