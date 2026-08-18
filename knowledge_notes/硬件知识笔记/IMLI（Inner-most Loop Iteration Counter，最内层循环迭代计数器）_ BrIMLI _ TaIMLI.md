## IMLI（Inner-most Loop Iteration Counter，最内层循环迭代计数器）/ BrIMLI / TaIMLI

术语解释
IMLI 是约 10-bit 的全局计数器，表示当前最内层循环的迭代号（global 量、不属于特定分支，比局部历史更易做推测性管理），作为 SC 组件的索引/输入捕获"迭代号 ↔ 分支结果"相关，是 TAGE-SC 中 sI/sIM 组件的基础。BrIMLI/TaIMLI 是 Seznec 2024 提出的稳健计数变体，处理编译器优化后的复杂循环结构。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：(1) 起源：Seznec、San Miguel、Albericio 提出 IMLI（MICRO 2015，"The inner most loop iteration counter: a new dimension in branch history"，DOI 10.1145/2830772.2830831），作为分支历史的"新维度"——WH（Wormhole）预测器利用多维循环相关但实现难（长推测性局部历史、要求恒定迭代循环），IMLI 以简单可推测管理的全局计数器替代；(2) 定义：最内层循环迭代号，在取指时高效维护，可部分替代局部历史预测循环退出分支——局部历史对多数程序收益小、IMLI 对特定程序收益大；(3) 两种组件：IMLI-SIC（Same Iteration Correlation）与 IMLI-OH（Outer History），可加进 TAGE 族或感知机预测器（如 TAGE-SC-L 的 sI IMLI-SIC、sIM IMLI-OH）；(4) 稳健计数变体（Seznec TAGE engineering cookbook，2024）：编译器可能把循环转换得没有"经典向后跳转"结构，TaIMLI 用"taken 向后分支指向与上一 taken 向后分支相同区域"来计数（处理存在 untaken 向后分支的编译输出），BrIMLI 用分支自身区域而非目标区域计数（Figure 9 两种编译输出的对比场景）；(5) RUNLTS 的发现：IMLI 对"顺序访问内容逐代微变的数组的循环"特别有效——数组变化使历史中相应部分失效、历史预测反复失效，而 IMLI 只依赖迭代号，不变迭代上的学习持续有效；循环含 hard-to-predict 分支时同样稳健（此类分支污染全局历史、局部历史也被失效）。
- RUNLTS 的增强：(a) UT 判定 IMLI useful 时 gain 从 2 提到 3（IMLI 在有效场景比 TAGE 更可靠）；(b) 同时 WT 更新步长×3（IMLI 条目易受邻近 hard-to-predict 分支别名攻击，加大步长抗腐蚀、方向变化时快速适应）；(c) UT 条目从 8 扩到 256（预测/训练都加强后需更精细区分目标 PC、输出常大于其他组件）。贡献：IMLI tweaks 平均降低 MPKI 0.74%，对 int_{1,2,21}_trace 收益最大。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件位置：SC 内的组件（sI IMLI-SIC 2.464 KiB、baseline TAGE-SC-L 中另有 sIM IMLI-OH 1.819 KiB，RUNLTS 移除 sIM 并把容量并入其他组件）。运转流程例子（逐代微变数组循环，Figure 9a）：fetch 维护 IMLI 计数器，每次迭代 +1 → 循环退出分支 B 预测时：SC 的 sI 组件按 (PC, IMLI 迭代号) 哈希访问权重表求和 → UT 判定 IMLI 有用 → gain 3 放大 → 参与 SC 总输出 → 循环内数组元素 a[i].valid/m<K 等分支即使历史被污染，迭代号维度仍给出稳定预测（学习在不变迭代上保持有效）。
- 例子（BrIMLI/TaIMLI 场景，Figure 9b/c）：编译器把循环编译成含 untaken 向后分支（图 b，TaIMLI 用"taken 向后分支目标与上一 taken 同区域"计数正确）或用分支自身区域（图 c，BrIMLI 计数正确），而原始 IMLI 依赖简单向后分支计数可能失败。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：取指单元维护的全局计数器（约 10-bit）+ 分支/目标区域比较逻辑（TaIMLI/BrIMLI 变体）+ SC 中的 IMLI 权重表（按 PC+IMLI 哈希索引的 hashed perceptron）。推测性管理：IMLI 是全局量、不绑定特定分支，flush 恢复比局部历史简单。使用场景：预测循环退出分支与数组内容逐代变化循环内的分支；常与 TAGE-SC 的 SC 结合（TAGE-SC-L 用 IMLI-SIC/IMLI-OH，RUNLTS 用增强版 IMLI-SIC）。
- Web 证据：IMLI 原始论文（MICRO-48，inria.hal.science/hal-01208347）、"Practical Multidimensional Branch Prediction"（IEEE Micro 2016，hal-01330510）。

涉及论文标题：
- RUNLTS Branch Prediction with Register-Value Correlations and Hierarchical Table Orchestration
