## Alecto（预取器选择与动态请求分配算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Alecto（Li、Zhang、Ren、Xie，HPCA 2025，"Integrating Prefetcher Selection with Dynamic Request Allocation Improves Prefetching Efficiency"，arXiv:2503.19390）是面向多预取器协作的预取器选择算法：每个 demand 请求先送 Alecto，由 Allocation Table 判定该请求适合交给哪个预取器，再路由到对应预取器做训练与预取，而非让所有预取器都看到所有请求。它用三态细粒度状态（UI 未识别/IA 识别为高效/IB 识别为低效并被临时屏蔽）+ M/N 个子状态分级调节预取度，用 Sample Table 与 Sandbox Table 收集准确率、demand 计数、dead 计数驱动状态迁移，Sandbox Table 兼作重复预取请求过滤器，并对时序预取器专门过滤 demand 请求以优化元数据存储（Web 证据：单核/八核分别超 RL 选择算法 Bandit 2.76%/7.56%，预取器表访问能耗降 48%，存储 <1KB，gem5 Intel Skylake-like 配置评估）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Alecto 在缓存层级中的运转流程（ICP 论文的 baseline 配置）：ICP 论文用 IPCP 的 stream、stride、spatial 三个基本预取器部署在 L1 缓存，由 Alecto 作为"预取器选择算法"协调它们——模拟商用 CPU（Arm Cortex X2 风格）的真实预取环境。demand 请求 → Alecto 的 Allocation Table 按请求（PC/地址）判定归属预取器 → 路由到对应预取器训练并产生预取 → 无效/低效预取被屏蔽、高效预取提升预取度。ICP 论文的动机正是建立在此 baseline 之上：这些基本预取器（即便经 Alecto 协调）只能捕获规则访存模式，对不规则访存无能为力——式(1) 用"在带基本预取器的系统中仍 demand miss 最多的 PC"来定位这类不规则 PC（PC_suc），即把 Alecto/IPCP 覆盖不到的残差作为不规则访存信号。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：Allocation Table（每指令多状态）+ Sample Table/Sandbox Table（运行时指标收集）+ 状态迁移逻辑，gem5 模拟器实现（作者主页 zhiyaoxie.com/files/HPCA25_Alecto.pdf 提供全文）；调度与 IPCP 相同的预取器集合（Arm Neoverse V2 风格：GS stream、CS stride、PMP spatial），与 Bandit、Triangel 对比，负载覆盖 SPEC CPU2006/2017、PARSEC 3.0、Ligra。使用场景：作为"多预取器协调/SOTA 预取环境" baseline（ICP 用它构造带基本预取器的真实基线系统，评估不规则预取器在其上的增量收益）；也是 ICP 论文中"basic-prefetcher-friendly（PC_pre^f）"分类的参照——能被基本预取器有效服务（Cov≥θ_cov）的 PC 走预取触发路径提升及时性，不能的（PC_pre^nf）只靠 demand 触发保证准确。

涉及论文标题：
- ICP: Exploiting Instruction Correlation for Prefetching Irregular Memory Accesses
