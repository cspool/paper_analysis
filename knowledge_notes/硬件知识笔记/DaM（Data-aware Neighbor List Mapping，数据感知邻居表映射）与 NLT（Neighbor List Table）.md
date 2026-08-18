## DaM（Data-aware Neighbor List Mapping，数据感知邻居表映射）与 NLT（Neighbor List Table）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DaM 是 NASZIP 的数据放置策略：把邻居表按"与对应向量同 sub-channel 共置"的方式切分存储，使每个 sub-channel 能独立取邻居并算本地向量的距离，消除跨 sub-channel 通信（跨通道访问必须经处理器中转、代价高），同时把邻居检索从 CPU 卸载到 NDP。NLT（Neighbor List Table）是管理变长邻居表的索引结构：记录每条邻居表的长度与内存地址，使变长分区表可高效索引。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DaM 运转流程（图 12，6 节点例子）：向量 1 的邻居 {2,3,6} 中 2、3 存于 sub-channel 0、6 存于 sub-channel 1 → 邻居表按此切分成两份分别放在两个 sub-channel → CPU 请求遍历向量 1 邻居时，sub-channel 0 并行处理 2、3、sub-channel 1 独立处理 6，无跨通道中转。每个 sub-channel 内存中放 NLT 表（图 12b）记录每条分区邻居表的长度与地址；检索路径：控制器查 LNC-T（NLT 的缓存）→ 命中/取 NLT 条目 → 按地址取邻居表（LNC-D）→ 得到本地邻居节点 ID 列表 → 下发算距。效果：跨通道通信从 naive NDP 的主要开销降为 0，邻居查找并行化且 CPU 侧 31.7% 的串行开销被卸载。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：离线把 HNSW 图按向量所在 sub-channel 重新切分布局（DaM），NLT 随布局生成；硬件上 LNC-T 缓存 NLT 条目（4B/条、8KB 全相联）。使用：配合 FEE-sPCA/Dfloat 的向量数据布局（每向量完整落在一个 sub-channel、维间 4 device 交叉），实现"邻居表+向量数据同驻本地"的访存局部化。效果量化：非距离侧延迟从 53.42%（ANSMET）降至 36.54%（DaM）、21.08%（+LNC）。开源实现见 NasZip 仓库 preprocess_idx/ 与 simulate/。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
