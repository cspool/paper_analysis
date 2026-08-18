## TDS-CSR Table（TDS 稀疏结构片上表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TDS-CSR Table 是 TAGT 片内（512KB）以 CSR（Compressed Sparse Row）格式组织 TDS 稀疏图结构的硬件表：TCU 构造完成后，表中含目标顶点、其偏移（offsets）与完整关联顶点 ID 及特征列表（original/fusion/association 边）。CSR 是图处理的标准稀疏存储格式（顶点偏移数组 + 邻居 ID 数组），TAGT 用它把每个目标顶点的 TDS 1-hop 邻域紧凑编码，供 Task Dispatcher 生成紧凑任务描述符（Partition ID、Target ID、Associated List Pointer、Count）与 FAU 取关联特征。
- 核心作用：(1) 作为 MOU 去重的查找表——共享 fusion 祖先 ID 可预计算，MOU 查表跳过已生成的 fusion 顶点，消除冗余计算；(2) 作为 Task Dispatcher 合并的依据——检查多个任务的 Associated ID 列表（经指针）找出共享 ID，合并打包；(3) 紧凑元数据替代全特征向量，减少 Task FIFO 流量。

从硬件架构角度拆解术语，比如术语在硬件架构中发挥作用的流程例子。通过联网搜索让回答具体和精准。
- 流程：TCU 的 FUU 生成叶子嵌入写入 TDS-CSR Table 并流式送 MOU → MOU 查表去重计算 fusion 顶点并更新表（目标顶点、offsets、关联 ID/特征）→ Task FIFO 从表生成紧凑描述符（指针指向关联列表）→ Dispatcher 经指针读关联列表、识别跨任务共享 ID → 合并打包发 GTPU。TAGT 配置 512KB TDS-CSR Table（Table V 片内存储的一部分）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：片上 SRAM（512KB），CSR 布局由 TCU 控制逻辑动态组装；是 TAGT 专用结构（非通用模拟器/软件库）。
- 使用：作为 TDS 原生执行表示贯穿"构造→去重→派发→注意力取数"全链路，配合 DDLM 与 FAU 实现硬件上的 O(N log N) 稀疏注意力；对比 baseline 加速器（FlowGNN/MEGA/BingoGCN）需自行处理 O(N²) 全对注意力数据，TAGT 以紧凑 TDS-CSR 大幅减少 off-chip 流量。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
