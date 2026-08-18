## 硬件 Random Forest Phase Predictor（阶段预测器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PhaseWeave 的硬件阶段预测器是一个片上轻量模块，用 Random Forest（RF）分类器在每 100µs epoch 末预测每个线程的下一执行阶段（4 类：compute/memory/network/low-power），完全透明于应用（只用硬件计数器 + 应用无关的系统调用分类频率）。动机：软件 RF 每次推理 40-250µs（加 PMU 采集 2-5µs、上下文切换 4-6µs），与 phase 时长同量级且落在关键路径，吞吐降 >20%；硬件实现 <100 cycles、占 core 面积 0.02%、每 epoch 仅 75 次比较，异步于应用核、不在关键路径。RF 在候选方法（阈值/聚类/HMM/Bandits）中准确率最高（>90%，未训练负载上 91%），存储仅 8KB。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
微架构（图 10）：训练好的模型存在紧凑 SRAM "Decision Storage" 中——每条目编码一条决策规则：4-bit feature id（选 IPC/cache MPKI/branch MPKI/TLB MPKI/系统调用频率等硬件或运行时指标）+ 16-bit 比较阈值 + 12-bit 左右子节点索引；叶子条目存 2-bit phase label；1 个控制位区分内部/叶子节点。在线推理流程：epoch 末 feature sampler 聚合计数器更新本地 feature buffer → traversal engine 并行遍历全部 15 棵树（每树深度 5，加载特征值、根到叶子逐层比较阈值、按左右索引下行，固定 5 步）→ voting unit 累积各类 phase 票数选最高票为最终预测。预测器 15 棵树 × 5 步遍历并行比较，实现微秒级（亚 100 cycle）离关键路径推理。训练离线完成（WDLBench 微基准灵敏度扫描打标），模型参数 boot 时装载，全应用共用、无需在线重训（phase 类别宽且稳定、硬件级指示跨负载一致）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：RF 可高效硬件化（论文引 [19] 硬件 RF 实现），每核一个预测器（或小核组共享以省面积功耗）；特征来源是 PMU（performance monitoring unit）计数器与系统调用分类计数，无需用户代码。使用：预测结果喂给收益感知迁移调度器（Algorithm 1）决定是否/迁往哪个 chiplet。泛化性证据：用 DCPerf 训练后在未参与的 DeathStarBench 上平均 91% 准确率（微基准学到的 phase 规律泛化到真实微服务）；RF 对短 phase 重叠噪声鲁棒（HMM/Contextual Bandit 对噪声敏感、Multi-Armed Bandit 纯分布驱动失效）。对照意义：为"phase 检测必须硬件化"提供了量化依据——软件预测每请求多出 10ms+（AdSim 250、Django 390 epoch/请求）延迟。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
