## 域专用加速器（DSA，Domain-Specific Accelerator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
域专用加速器（DSA）是面向特定计算域（矩阵运算、数据搬运、加密等）设计的专用硬件加速器，以牺牲通用性换取比通用 CPU/GPU 高得多的能效与吞吐。注意与 Intel Data Streaming Accelerator（Intel DSA，见本库 `Intel DSA / IAA` 条目，面向数据搬运的特定产品）区分：此处 DSA 指架构设计中"域专用加速器"这一类通用概念。RHODES 在异构 SoC 设计空间探索中把 DSA 作为可选加速组件：论文按先前工作 [55] 把 DSA 建模为"4× GPU 的能效"，d_m^n 表示 m 个 DSA、每个含 n 个处理单元（PE），用于 workload level parallelism（WLP）分析（§V-B）——即同等性能下 DSA 的功耗/碳开销约为 GPU 的 1/4。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 的 WLP 案例中，DSA 与 CPU 核、GPU SM、HBM 主存共同构成异构 SoC 设计空间：工作量（Rodinia 负载）的 compute 阶段可选择映射到 CPU、GPU 或 DSA（二进制选择变量），各选项有独立执行时间/功耗/面积/制造碳参数。DSA 的 4× GPU 能效使碳约束下能容纳更多计算资源（同样 tC 预算下 DSA 方案可比纯 GPU 方案配置更多 PE），优化器在 tC 与性能约束下联合选择 CPU 核数、GPU SM×频率、DSA 数×PE 数的最优组合。运转流程：定义各组件候选配置集 → 用 HILP profiling 数据标定每配置执行时间与功耗 → 编码 MILP 约束（执行时间/功耗/面积/tC）→ Gurobi 求解 → 输出含 DSA 数量的最优 SoC 配置（RHODES robust 设计在 Monte Carlo 评估的 tC 的 0.951–1.147× 内，vs nominal 低估 1.17–1.55×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DSA 作为 SoC 上的专用 IP 块（如矩阵乘引擎、专用数据流加速器），面积/功耗/性能由其 PE 阵列规模与技术节点决定；RHODES 用"m 个 DSA × n 个 PE"的参数化描述并给出 4× GPU 能效的抽象标定。使用：设计师在早期设计阶段把 DSA 候选加入配置空间，与 CPU/GPU/HBM 一起在碳-性能-面积约束下联合优化；其制造碳参数（FPW/GPW/MPW 区间）与其他 die 一样纳入鲁棒 tC 约束。开源：RHODES GitHub（https://github.com/mariamelgamal/RHODES，仅 README）。相关：HILP [55]（HPCA 2025）对 DSA 能效的建模先例、Neoscope [56] 的 SoC 工作量流失弹性分析。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
