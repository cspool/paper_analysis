## LNC（Local Neighbor Cache，局部邻居缓存：LNC-T / LNC-D）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LNC 是 NASZIP 在 NDP 上利用 ANNS 查询局部性（相似/重复查询反复访问相同图节点）设计的两级邻居缓存，由 LNC-T（Local Neighbor Cache for Table）与 LNC-D（Local Neighbor Cache for Data）组成：LNC-T 缓存邻居表索引（NLT 条目）、功能类似 TLB；LNC-D 缓存邻居表内容、功能类似数据缓存。两者配合减少冗余邻居表访问、加速邻居检索并配合预取。配置：LNC-T 8KB 全相联、LNC-D 256KB 8-way 组相联、64B cache line（=sub-channel burst 大小）；LNC-D 容量每 sub-channel 256KB。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LNC 工作流（图 13，计算向量 i 的邻居距离时）：① 控制器查 LNC-T 中 i 的 NLT 条目 → miss → ② 从内存取 NLT 条目 → ③ 插入 LNC-T → ④ 从 LNC-T 读条目得邻居表地址 → ⑤ 用地址查 LNC-D（本例命中）→ ⑥ 按长度信息（如 3）取出本地邻居节点 c/d/e 并下发算距请求。Tag 设计差异：NLT 条目 4 字节（3 字节起始地址+1 字节长度），LNC-T 每 line 存 16 条、tag 只记首条目 ID；LNC-D 因邻居表跨 sub-channel 长度不一、tag 记录段起止节点 ID。命中率行为（Fig.21a）：LNC-D 越大命中率越高；efSearch 增大（搜索范围广、访问节点多样）命中率下降；超过 efSearch>50 后热邻居表基本驻留、命中率收敛。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 NDP 硬件模块与 VPE、共享优先队列、控制器一起封装进 DB 芯片（LNC-D 489.6K μm²、LNC-T 37.5K μm²，28nm 综合）。使用：与邻居表预取配合——预取失败的内容留在 LNC 中仍可被后续访问复用，使预取开销很小；非距离侧延迟由 DaM 与 LNC 从 53.42%（ANSMET）降到 36.54% 与 21.08%（论文 Fig.25）。开销评估见硬件架构实验条目。开源实现见 NasZip 仓库 simulate/。

涉及论文标题：
- NasZip Software and Hardware Co-design to Accelerate Approximate Nearest Neighbor Search with DIMM-based Near-Data Processing
