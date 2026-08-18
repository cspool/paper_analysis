## 语义感知动态 tile 调度器（Semantics-Aware Dynamic Tile Scheduler，硬件调度器）

术语解释
集成在 AI 加速器每核内的硬件调度器（RTL 实现，随 Epoch 芯片流片），消费 TISA 指令的语义字段，在 tensor（ME）/vector（VE）/DMA（DE）三类异构单元间动态重排 tile、消解结构性争用与数据依赖、实现跨算子与跨迭代的运行时重叠。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 论文机制（Section V + VII）：四步协同——① Semantic Routing：解析 TISA 指令的 OpType/UnitMap/依赖元数据，路由到目标单元的等待队列 WQ（每单元独立语义队列对，防止异构单元间无关阻塞）；② Dependency Resolution：周期性地从每个 WQ 选就绪窗口 W，对候选与单元 in-flight 语义表 F_u 做语义冲突检测（Algorithm 2），通过依赖与资源检查者提升到 issue queue IQ；③ Adaptive Issue：依赖清除后乱序发射到硬件执行流水线；④ Feedback：完成时退休 F_u 条目、通知依赖指令、按观测到的竞争/延迟自适应更新每单元调度优先级。
- 五级微架构：Reception（指令解码）→ Routing（路由到各单元 WQ）→ Dependency check（窗口匹配）→ Issue（无冲突指令 WQ→IQ）→ Dispatch（IQ 到执行单元）。指令 run-to-complete、非抢占，调度决策只在 tile 边界（tile 通常 >10³ 运算），控制开销低（7~9 cycles/dispatch）。
- 调度循环复杂度 O(U·W·|F|max)，典型 W≤8、|F_u|≤16 时近似 O(U)/cycle，优于集中式 Tomasulo 的 O(N²) 全局比较。RTL 综合：W=8 时 1.5M gates（0.25 mm²，占每核面积 1.5%，100 mW）；W=256 时 6.8M gates（1.13 mm²，300 mW），面积亚二次方增长（4.5× 面积换 32× 条目）靠对数 CAM 结构与 W≥32 流水仲裁；功耗 <0.3% 核功耗（dispatch 稀疏，~5% slots/cycle）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 每个加速器核集成一个调度器（Table IV），多核时每核本地 TISA 调度器独立运行（各自 F_u），核间同步用共享 SRAM bank 更新触发的轻量 NoC 信号；核间负载均衡在 kernel 调用间由软件做，tile 调度器内不做。
- 运转流程例子（FA3，图 2(c/e)）：静态流水把 M0^i/S^i/M1^i 固定排成双阶段或三阶段并强制迭代间屏障，S^i 与 M0^{i+1} 数据独立且资源不冲突却被序列化；动态调度器观测到向量单元空闲且 M0^{i+1} 输入就绪，就在 S^i 执行期间把 M0^{i+1} 发射到矩阵单元，同时 S^{i+1} 可与 M1^i 并发——相比静态双阶段/三阶段累加节省 E0+E2 或 E1+E3 的延迟，且能自适应 DMA backpressure、bank 冲突、热节流等运行时变动。
- 硬件代价：论文在 Epoch 真硅片（W=8）上验证，Dynamic vs Naive 1.52–1.92×、vs 最强静态 tile 流水再高 1.14–1.63×；Accumulated Overlap Score（DM/DV/MV/DMV 四类跨单元重叠周期累加）Dynamic 达 Naive 的 2.60–32.64×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 综合实现（每核一个），直接消费编译二进制中嵌入的 TISA 元数据做调度决策，无需软件屏障。论文明确该功能不能由软件实现：软件运行时依赖检查需微秒级（100~1000× 慢于硬件的 7~9 cycles），无法支撑 tile 级纳秒级机会性重叠。
- 使用：作为 Epoch 的软件-硬件契约，调度器按 Algorithm 1 持续运行（reception buffer 非空则弹指令、提取语义、自适应选单元入 WQ；每单元窗口内做冲突检测与资源分配；完成时更新状态并触发依赖）。可移植：对其他异构加速器只需添加硬件调度器与 TISA；GPU 上可加语义感知协调器叠在 warp/CTA 调度器之上管理跨单元依赖。
- 开源情况：论文未给出调度器 RTL/代码开源链接，联网搜索未发现公开仓库。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
