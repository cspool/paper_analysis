## 数据驱动加载与合并引擎（TDL / MAPE / TCU / FUU / MOU，含 Task Dispatcher）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据驱动加载与合并引擎（Data-driven Loading and Merging，DDLM）是 TAGT 中负责"取数→构造 TDS→派发任务"的硬件子系统，由 Topology Data Loader (TDL)、TDS Construction Unit (TCU)、Task Dispatcher 组成：(1) TDL 的 Memory Access Pipeline Engine (MAPE) 用多个 6 级硬件流水线分阶段取数——Fetch_Root/Fetch_Offsets/Fetch_Neighbors 从 CSR 结构定位目标顶点邻域，Fetch_Features/Fetch_Weight/Fetch_Coding 取特征/权重/结构嵌入；复制 Fetch_Neighbors 与 Fetch_Features 单元掩盖 HBM 延迟、防流水停顿，数据转 Construction FIFO Buffer。(2) TCU 含 Feature Update Unit（FUU：同步取原始特征与编码向量、轻量拼接、线性投影生成基层叶子嵌入，对 SE/PE 语义无关）与 Merge Operation Unit（MOU：多级并行加法树高扇入聚合，递归计算全部 fusion/associated 顶点嵌入），并内置硬件去重——查 TDS-CSR Table 跳过已算过的共享 fusion 祖先只算一次；TCU 控制逻辑组装 TDS 稀疏图结构（original/fusion/association 边）写入 TDS-CSR Table，向 Task FIFO 填紧凑任务描述符（Partition ID、Target ID、Associated List Pointer、Count）。(3) Task Dispatcher 检查多个 pending task 的 Associated ID 列表，合并共享 Associated ID 的任务打包成单一请求发往 GTPU，一次 dispatch 服务多次注意力计算（共享关联特征跨目标顶点复用）。
- 动机：GPP 上 TDS 构造的软件 runtime overhead 占 69.8%–86.1% 执行时间（TAGT-S），且动态稀疏 gather 不规则、共享 fusion 祖先导致冗余计算与取数。DDLM 把 TDS 构造做成硬件流水并去重，将软件开销转为并行计算。

从硬件架构角度拆解术语，比如术语在硬件架构中发挥作用的流程例子。通过联网搜索让回答具体和精准。
- 运转流程（Fig.7）：partition D_0 到达 → TDL 的 MAPE 6 级流水取邻域/特征/权重/编码（复制 Fetch_Neighbors/Fetch_Features 掩延迟）→ Construction FIFO Buffer → TCU：FUU 生成叶子嵌入（写 TDS-CSR Table 并流式送 MOU）→ MOU 多级并行加法树自底向上聚合 fusion 顶点（查表去重共享祖先）→ 组装 TDS 结构入 TDS-CSR Table → Task FIFO 填紧凑描述符 → Task Dispatcher 按共享 Associated ID 合并打包 → GTPU/FAU 执行注意力（见 GTPU/FAU 与 SCU 条目）。
- 消融效果：w/o DDLM → 平均 4.41× 性能损失，DDLM 贡献 71.38% 总性能收益。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 综合到 Alveo U280（Vivado 2019.1，280MHz）。规模：8 FUU、8 MOU、1 TDL（MAPE 多 6 级流水）、1 Task Dispatcher；FIFO/Buffer：Construction FIFO、Task FIFO 128KB、TDS-CSR Table 512KB。m（合并基数）经片上配置寄存器编程。
- 使用：作为 TAGT 的前端数据通路，负责把 HBM 中 CSR 图数据+特征+编码转成紧凑 TDS 任务流喂给 GTPU；去重与合并机制最小化 off-chip 访问（off-chip 流量降 42.1%–81.6%）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
