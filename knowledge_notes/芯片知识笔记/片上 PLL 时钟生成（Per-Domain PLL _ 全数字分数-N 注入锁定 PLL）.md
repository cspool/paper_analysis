## 片上 PLL 时钟生成（Per-Domain PLL / 全数字分数-N 注入锁定 PLL）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 锁相环（Phase-Locked Loop, PLL）是芯片内的时钟发生器：以参考时钟为基准，用反馈环路把压控振荡器（VCO）锁定到目标频率并输出稳定时钟。每个独立时钟域通常需要一个 PLL（或多个域共用 PLL + 分频器）。空间 DVFS 增加独立电压域数量的同时往往增加独立时钟域数量，每域需自己的 PLL——论文把"每个新增电压域引入一个全新独立 PLL 时钟域"作为保守上界假设，评估时钟生成面积开销。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片设计中的运转流程：DVFS 域数 N → 每域一个 PLL 实例 → 面积 ΔA_PLL(N) = A_PLL,unit × (N−1)。论文直接采用一颗 5nm FinFET fully-synthesizable fractional-N injection-locked PLL（专为 manycore 系统设计，报告面积 0.0036 mm² [28]）作为 per-domain PLL 面积。作用在总模型：ΔA_tot(N) = ΔA_reg(N) + ΔA_LS(N) + ΔA_PLL(N)，与 DLDO 与边界同步叠加。per-SM 148 域时 PLL 项 0.0036×147 ≈ 0.53 mm²，占总 die 0.14% 量级。论文注明这是保守上界：实际 GPU 实现可能用单 PLL + 多分频器服务相邻时钟域，进一步降低成本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：现代处理器/加速器普遍用层次化时钟树 + 分频器为多域提供时钟，全数字 PLL（ADPLL）可综合、易迁移（如 [28] 的 fractional-N injection-locked PLL）。PowerWeave 把它作为空间 DVFS 三项面积成本之一（Table IV：Clock Generation 0.0036 mm²/域，0.00014% die），结论是时钟生成开销相对边界同步很小，不是空间 DVFS 的成本瓶颈。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
