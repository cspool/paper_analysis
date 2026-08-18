## Hashed Perceptron 与 GEHL 感知机分支预测器

术语解释
Hashed perceptron 是用多个按不同哈希索引的权重表做分支方向预测的感知机类机制；GEHL（GEometric History Length，Seznec ISCA'05）是其用几何级数历史长度索引的实现，统计校正器（SC）的基本构件。预测 = 各表取回权重的和，符号决定方向。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑链：(1) 感知机预测器（Jiménez & Lin，HPCA 2001）：为每个分支关联一组 8-bit 计数器/权重，输入为全局历史向量（+1/-1），预测 = 权重×输入累加和的符号；(2) 哈希化：把索引换成对（PC + 分支历史不同长度的哈希），同一张权重表可被所有分支共享，解决"每分支专属权重"容量爆炸；(3) GEHL/O-GEHL（Seznec）：用 M=4~12 张权重表，历史长度呈几何级数（如 {0,2,4,8,16,32,64,128}），Sum = M/2 + Σ C(i)，Sum≥0 预测 taken，否则 not-taken；(4) 训练规则：只在 (a) 预测错误，或 (b) 预测正确但加权和绝对值低于阈值时更新权重（阈值训练，O-GEHL 用自适应阈值）；正确且 |sum| 超阈值时不更新——防止过拟合、允许别名到同一 entry 的其他分支"偷取"权重；(5) 相比 TAGE 的差异：GEHL 每次更新所有历史长度表，TAGE 只更新最长匹配表——这一差异在 RUNLTS 中被利用（见调用栈历史条目：数组逐代变化的循环里 GEHL 短历史表保持有效、TAGE 长历史表整体失效）。
- 在 RUNLTS 中：SC 的每个组件（sB Bias、sG、sP、sL/sS/sT、sI、sC）都是 hashed perceptron/GEHL 型，权重表 + UT usefulness-tracking 表（仅按 PC 索引的饱和计数器，为正时该组件加权和乘固定 gain）；RBias 也是感知机型（WT + UT），但输入是寄存器值 digest 而非分支历史。
- 别名（aliasing）：多个分支共享同一表项导致权重互相干扰；gskew（Michaud 等，ISCA'97）用多个不同哈希函数访问同一表取多数/求和以降低破坏性干扰——RUNLTS 的 RBias UT/WT 都采用 gskew 式 3-hash 组织。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件中位于 fetch 单元旁的分支预测通路，与 TAGE 并行或级联（SC 在 TAGE 预测后修正）。运转流程例子（RUNLTS 的 SC 修正一次预测）：分支 PC 到达 → 各组件按各自索引函数（PC ⊕ 不同长度历史哈希）并行访问多张权重表，sB Bias 组件额外输入 TAGE 预测方向/置信度 → 每组件检索权重求和、乘各自 gain（UT 判定 useful 时放大）→ SC 总加权和与 TAGE 预测方向比较：同向则沿用 TAGE、异向且幅度超阈值则翻转 → 若最终预测错误，各组件按规则训练（错误或低置信度时更新命中表的权重，沿预测方向 ±1）→ 下一周期权重就绪。
- 例子（感知机训练更新）：预测错误时，对每个访问过的表项 w_i 按 w_i ← w_i + (实际方向×输入特征) 更新（特征 +1/-1），阈值 T 在 |sum| < T 时也触发同样更新；O-GEHL 的自适应阈值在"误预测触发更新 vs 低置信度触发更新"之间动态调整 T。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：多张权重表（SRAM 阵列，每项 4-8 bit 饱和有符号计数器）+ 哈希网络（对 PC 与各长度历史做 XOR 折叠，O-GEHL 用单级 3-entry XOR 取 ≤33 bit）+ 加法树（各表取回值累加）+ 比较器（符号/阈值判定）+ 更新逻辑。在 CBP simulator 中表现为 cond_branch_predictor 接口内的权重表读写；权重表容量由 192 KiB 总预算约束（如 RUNLTS 中 sG 3.011 KiB、sL 5.069 KiB 等）。使用场景：作为 TAGE-SC 的统计修正层捕获长期统计偏置（TAGE 的 3-bit 计数器表达不了细微长期偏置），以及作为独立预测器（Multiperspective Perceptron，CBP-2016 冠军，Jiménez）。
- Web 证据：Seznec 的 GEHL 论文（O-GEHL 案例研究）、Jiménez CBP-2016 Multiperspective Perceptron（https://jilp.org/cbp2016/paper/DanielJimenez1.pdf）、arxiv 1804.00261 综述。

涉及论文标题：
- RUNLTS Branch Prediction with Register-Value Correlations and Hierarchical Table Orchestration
