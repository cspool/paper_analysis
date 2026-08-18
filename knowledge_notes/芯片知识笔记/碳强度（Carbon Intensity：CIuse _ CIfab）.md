## 碳强度（Carbon Intensity：CIuse / CIfab）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
碳强度 CI 是单位电能对应的 CO2e 排放（kg CO2e/kWh），取决于发电能源结构（煤/气/可再生能源占比）。RHODES 区分两个碳强度：CI_fab（fab 用电碳强度，决定制造碳）与 CI_use（使用期电网碳强度，决定运营碳）。二者归类不同：CI_use 是 Data Value（DV）不确定性——CAISO 等电网数据海量（小时粒度、720 点/月、可实时追踪），有高方差但分布可估计，用概率鲁棒（U_PRO）不确定性集建模；CI_fab 视数据可得性可作 DV（用 CAISO 等电网数据）或 DA（如 TSMC 公开可持续报告仅少数数据点，0.494–0.5849 kg CO2e/kWh vs 台湾电网 0.474–0.562）处理，RHODES 两种均支持。化石电网与太阳能电网碳强度差 19.5×。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 中 CI_use 通过 U_PRO 集编码进运营碳：用 CAISO 2022–2024 小时级 CI_use 样本（含时间与能源来源波动）构建 log-survival 可靠函数与 U_PRO 不确定性集（Γ=1），再与 t_operational 合并为线性化参数 C_op=CI_use·t_operational，乘各阶段功耗与执行时间构成运营碳约束项。CIfab 与 EPW 合并为 FPW=CIfab·EPW 进入制造碳。运转流程：工作量 setup/compute/teardown 三阶段的运营碳各自约束"对 U_PRO 内所有 CI_use 实现均不超 tC_max"。Monte Carlo 验证时 CI_fab 与 CI_use 从有界正态分布采样（下限 0.041 kg CO2e/kWh 对应太阳能）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
数据源：CI_use 用 CAISO（https://www.caiso.com/TodaysOutlook/Pages/emissions.html）与 Electricity Maps（https://app.electricitymaps.com/）；CI_fab 用 TSMC 可持续报告 [61]-[63]、[68] 与台湾能源局数据。使用：DV 场景把 CI 样本向量喂给 U_PRO 集（uCop 函数），DA 场景用区间集；设计者按数据可得性切换。意义：把"电网在系统生命周期内的波动"与"fab 用能来源的不确定性"显式纳入设计，使运营/制造碳约束在真实波动下仍成立（nominal 设计在未考虑运营时间/CI 不确定性时 tC 可差 1.7×）。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
