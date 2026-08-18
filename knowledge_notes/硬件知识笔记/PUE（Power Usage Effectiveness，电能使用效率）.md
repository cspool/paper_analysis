## PUE（Power Usage Effectiveness，电能使用效率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PUE（Power Usage Effectiveness）是数据中心能效指标，定义为设施总能耗与 IT 设备能耗之比：PUE = 总设施能耗 / IT 能耗。PUE=1.0 表示全部电能都供给 IT 设备（理想）；越高表示越多能量浪费在冷却、供电转换等基础设施开销上。现代现代数据中心普遍 1.1–1.3（Google 报告自有数据中心 1.09、行业平均 1.56；GLaM 训练时 TPU-v4 数据中心 PUE=1.11）。本论文（Rearchitecting the Datacenter Lifecycle for AI）把 PUE 纳入 TCO 模型（表 III）：能量 OpEx = IT 负载 × PUE × 电价，即冷却/供电开销通过 PUE 放大 IT 能耗进入成本；同时把 PUE 作为蒙特卡洛随机变量（Normal，均值=baseline，σ=0.05）建模其不确定性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PUE 在数据中心能量成本计算与冷却决策中的运转流程：
```
IT 负载功率 P_IT(t)（由 roofline goodput 决定的 GPU 供给 × 单机功耗）
→ 设施总能耗 = P_IT(t) × PUE（PUE 反映冷却效率与供电损耗）
→ 年能量 OpEx = P_IT × PUE × 8760h × 电价($20–40/MWh)
→ 冷却设计选择影响 PUE：air(高风扇/高 PUE) vs liquid(低 chiller 负载、低 PUE) vs hybrid
→ 论文结论：75/25 hybrid(高密度 rack 液冷 + 低密度 air) 使 TCO 降 9%（表 VI）
```
PUE 与冷却设计的耦合是 build 阶段的关键：air 冷却 PUE 较高但 CapEx 低，liquid 冷却 PUE 低但 CapEx/维护高，hybrid 在两者间取得平衡。框架把该权衡折算进 15 年 TCO：低 PUE 的长期能量节省 vs 高冷却 CapEx 的一次性投入。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/度量：PUE 通过智能电表（设施总入口 vs IT 回路）持续测量，Uptime Institute 等机构统计行业基准；设计阶段按冷却方案（air chiller/CRAC、液冷 cold plate/immersion、hybrid）预估。本论文在 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）中以 YAML 参数化 PUE（含 Monte Carlo 分布），能量 OpEx 模块按 `IT负载 × PUE × 电价 × 运行小时` 计算。相关论文语境：ParetoES 的 TCO 公式 $TCO = C_{hw} + (P/1000)·H(L,u)·PUE·c_e$ 同样以 PUE 放大功耗成本；Lit Silicon 论文报告 GPU 功率约占供给功率 50%、负载平均 75% TDP。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
