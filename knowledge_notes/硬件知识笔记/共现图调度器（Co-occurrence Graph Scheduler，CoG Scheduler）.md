## 共现图调度器（Co-occurrence Graph Scheduler，CoG Scheduler）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
共现图调度器是 TensorPrism 加速器中执行"图构建 → CoGTP 划分 → 任务分派"的专用硬件引擎（§VI-B，Figure 8），硬件实现 Algorithm 1 与式 6 目标函数。组成：权重计算单元（weight computation unit，Coordinate Parser 提取索引对、Dimension Pair Selector 选维度对、Hash-based Engine 去重、Index Pair Buffer（FIFO，可配深度如 256）缓冲共现对及指向原元素的元数据、维护共现计数）、图生成器（graph generator，从稀疏张量元数据构造 CSR 共现图）、成本分析器（cost analyzer，按式 6 评估分区质量，含 PE 分区索引缓冲）、加法器（Adder，聚合各分析器成本指标算整体调度质量分）。分区索引缓冲存操作到具体 PE 的最终映射。面积仅占加速器 1.9%（成本分析器占调度器面积 62.9%）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程（输入→输出）：稀疏张量元数据（COO）→ 权重计算单元扫描坐标、提取共现对并计数（去重）→ 图生成器构造 CSR 共现图 → 成本分析器评估式 6（复用奖 - cut 惩罚 - 均衡惩罚）→ 加法器汇总 → 若 ΔF>ε 触发 CoGTP 边界顶点迁移迭代 → 分区索引缓冲固化 PE 映射 → 经指令分派器把"分区+共现图元数据+稠密张量操作数"分发给 16 个 PE。它同时支撑式 5 DRAM 访问模型（GLB 容量 $M_{cap}$ 约束下选 tiling 因子 $M_t$）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：专用硬件流水线（权重计算→图生成→成本分析→分派），与任务控制器、指令分派器、DRAM 控制器协同。相比 baseline：TCP 编译期探索大量实现选择（预处理开销最高 25.4%）、GSpTC 执行期反应式匹配（串行依赖发现）；CoG Scheduler 把划分决策硬件化、静态预测（划分前统一分析所有维度索引交叠），预处理开销仅 +8.0%/6.7%/4.2%（vs SPADE/HotTiles/GSpTC）。使用场景：FROSTT 8 数据集 + LLaMA 注意力张量收缩请求的片上图构建与分区。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
