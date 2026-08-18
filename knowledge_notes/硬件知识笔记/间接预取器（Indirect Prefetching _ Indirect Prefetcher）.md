## 间接预取器（Indirect Prefetching / Indirect Prefetcher）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 间接预取器是面向嵌套数组访问（如 a[b[i]]，内层 b[i] 的取数结果作为外层 a[] 的索引）的一类硬件预取器。逻辑链：(1) 嵌套数组访问中，内层数组 b[i] 的访问通常有规则 stride（循环内 i 递增），外层 a[b[i]] 的地址则不规则；(2) 若能提前拿到 b[i+d] 的值，就能算出外层地址 a[b[i+d]] 并预取；(3) 因此间接预取器分两阶段：识别阶段——检测内层 striding load 及其数据与依赖 load 地址的相关性（indirection pair）；预取阶段——用 stride 预取器预测 b[i+d] 的完整地址、从预取的 cache line 精确取数，加上外层数组基址生成 a[b[i+d]] 预取请求。代表设计：IMP（MICRO 2015）、Prodigy（HPCA 2021）、DMP（Differential-Matching Prefetcher，HPCA 2024）、Tyche（TACO 2024）、APT-GET（EuroSys 2022）。ICP 论文指出其核心局限：识别与预取机制都耦合数组语义——假设 PC_inner 必须是 striding load（即便 Tyche 想捕获更一般的间接模式也保留该假设），一旦内层无规则 stride（如 if(cond) x=a[b[i]]）或模式超出嵌套数组（array-of-pointers、链表指针），间接预取器即失效；SPEC 上 ICP 平均识别 22 条相关性 vs Tyche 7 / DMP 6，且 DMP 的 6 条中 2 条无效。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 间接预取器在缓存层级中的运转流程（DMP，HPCA 2024，Web 证据 DOI 10.1109/HPCA57654.2024.00040）：集成在 L1 缓存（ICP 论文中 DMP/Tyche 均集成 L1）。识别阶段——用差分匹配（differential-matching）机制把索引流（index stream，b[i] 的取值序列）与其对应数据数组访问模式（a[] 的地址流）关联，发现 indirection pair (b[i], a[b[i]])；预取阶段——依赖 stride 预取器预测内层 load 的未来地址，从其预取 cache line 中精确提取索引值，用外层数组基址 + 索引算出外层地址，按动态自适应度（flexible prefetching degree）发预取请求，在覆盖与资源消耗（能耗）间权衡。对比间接预取器与 ICP 的差别：间接预取器必须用 stride 预取器预测 PC_inner 的"完整地址"才能从 line 精确取 offset 对应的值；ICP 的 Data Extractor 则允许从"任意基本预取器提前取回的 cache line"用历史 offset 概率推断取数，触发更早、更通用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：硬件表 + 相关逻辑（DMP 的差分匹配器、Tyche 的 indirect memory access 检测、IMP 的 indirect pattern 表），均为 gem5/ChampSim 类模拟器内实现，无统一开源；DMP 论文称其为低开销硬件预取器，通过自适应预取度控制能耗。使用场景：作为嵌套数组间接访存的 SOTA baseline（ICP 与 Tyche/DMP 对比即为其典型用法）；与时序预取器组合（DMP+Triangel）可达到与 ICP 相当的性能（25.61% vs 25.51%），但继承两者缺点：元数据存储大（Triangel 最高 1MB + 17.6KB，DMP 912B）、DRAM 流量高（DMP+Triangel 比 ICP 多 6.84%）、能耗高（比 ICP 高 4.9%）。

涉及论文标题：
- ICP: Exploiting Instruction Correlation for Prefetching Irregular Memory Accesses
