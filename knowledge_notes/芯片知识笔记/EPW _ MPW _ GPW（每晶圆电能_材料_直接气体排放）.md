## EPW / MPW / GPW（每晶圆电能/材料/直接气体排放）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EPW（Energy Per Wafer，每晶圆电能耗，kWh/cm²）、MPW（Materials Per Wafer，每晶圆材料排放，g CO2e/cm²）、GPW（Gases Per Wafer，每晶圆直接气体排放，g CO2e/cm²）是 IC 制造碳的三个核心分解参数：EPW 是 fab 制造工具的耗电（与晶圆吞吐、技术节点、工艺流、工具利用率相关），MPW 是材料采购产生的排放，GPW 是工艺过程直接排放的温室气体（经 abatement 处理后可削减）。三者与碳强度 CIfab 及芯片面积/良率共同构成制造碳：C_embodied=(CIfab·EPW+MPW+GPW)·A/Y。RHODES 把它们归类为 Data Availability（DA）不确定性参数：EPW 数据跨文献每节点最高 1.59× 差异（imec 2020 假设 100% 工具利用率 vs imec LCA 2023 的 70% 利用率+50% 空闲能耗）；GPW 在 ≥95% abatement 下每节点最高 3.76× 差异，而美国 fab 实际 abatement 仅 0–77%（vs 模型常用的 95–99%）；MPW 是最难量化的参数，很多工作省略或跨节点取常数，RHODES 用 ±5% 保守不确定性包裹名义值 500 g CO2e/cm²。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 中 EPW/MPW/GPW 以每晶圆 kg CO2e 值经 wafer 面积 A_w=70,685.83 mm² 归一化后，乘 die 面积构成制造碳约束项，且 CPU 核 die（c）、CPU I/O die（cIO，12nm）、GPU SM die（g，7nm）按技术节点各取不同区间：Table II 示例 FPW_c 565.5–888.0、FPW_cIO 380.6–587.3、FPW_g 565.5–904.3；GPW_c 65.7–247.1、GPW_cIO 52.5–141.2、GPW_g 65.7–247.1；MPW 三者在 318.1–388.8。为保持 MILP 线性，CIfab·EPW 合并为 FPW 作为新线性化变量（未知×未知=未知×常数）。运转流程：每个候选配置的制造碳 = Σ_die [(FPW+GPW+MPW)⊙A_die]·c/g 选择向量，且对 U_interval 中所有取值都须满足 tC 约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
数据来源：EPW/GPW 来自 imec DTCO/PPACE [27]、imec LCA [13]、SW Jones 300mm fab 碳模型 [39]、imec.netzero [35]；GPW abatement 数据来自 EPA FLIGHT（美国 fab 报告，0–77%）与 TSMC/Intel 报告；MPW 来自工艺流分析 [31]。使用：在 RHODES 中为每 die 指定 LB/UB 区间（U_interval 不确定性集），优化器在区间内所有实现下保证 tC 约束。Monte Carlo 验证时 EPW/GPW 用有界正态分布（非负拒绝重采样）、MPW 用 ±5% 均匀分布。意义：把制造参数不确定性显式纳入设计优化，避免 nominal 设计因真实变异违反碳约束（如忽略 GPW abatement 不确定性在运营碳主导时仍可移动鲁棒最优，tCDP 变化 1.02×）。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
