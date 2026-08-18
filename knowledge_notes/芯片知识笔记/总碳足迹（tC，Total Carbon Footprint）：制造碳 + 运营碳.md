## 总碳足迹（tC，Total Carbon Footprint）：制造碳 + 运营碳

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
总碳足迹 tC 是计算系统全生命周期（含制造与使用）产生的 CO2 当量（CO2e）排放总量，定义为制造碳（embodied carbon，C_embodied）与运营碳（operational carbon，C_operational）之和（论文 Eq.4）：tC = (CI_fab·EPW + MPW + GPW)·A/Y + ∫_0^{t_operational} CI_use(t)·P(t) dt。制造碳由晶圆制造的电能消耗（EPW）、材料获取（MPW）与工艺直接气体排放（GPW）三部分乘以碳强度（CI_fab）按芯片面积 A 与良率 Y 折算到单 die；运营碳由使用期电网碳强度 CI_use 对系统功耗 P 在运行时间 t_operational 上的积分。GHG Protocol 视角下这是 Scope 1/2/3 的汇总（见本库 `制造碳（Embodied / Manufacturing Carbon）` 条目）。RHODES 的核心创新是把 tC 公式里的不确定参数（EPW/MPW/GPW/CI_use/t_operational）显式编码进鲁棒优化，使 tC 约束对所有不确定性集内实现都成立。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 中 tC 作为优化约束参与 SoC 配置选择：每个候选配置（CPU 核数 c_n + GPU SM 数/频率 g_m^f）的 tC = CPU 核 die 制造碳 + CPU I/O die 制造碳 + GPU die 制造碳 + 各阶段运营碳。制造碳按 die 分解：[(FPW+GPW+MPW)⊙A_c]^T·c 等（FPW=CIfab·EPW 线性化合并项），面积 A 来自 AMD EPYC 7453 核（10.125 mm²@7nm）与 NVIDIA A100 SM（7.65 mm²@7nm）；运营碳按工作量三阶段分解：C_op,c,s（setup 于 CPU）、C_op,c,k / C_op,g,k（compute 于 CPU 或 GPU）、C_op,c,td（teardown 于 CPU），每项 = CI_use·t_operational·功耗·执行时间 的线性化。运转流程例子：pathfinder 负载在 tC≤10 kg CO2e 约束下，nominal 优化选 c3g20（MC 评估 p95 tC=14.00 kg、54.4% 样本违反约束），RHODES robust 优化选 c1g28（p95=9.31 kg、违反率 ≤1.1%）。计入主存 HBM 的 tC 后，同约束下配置降配（c1g53→c1g19），忽略内存 tC 会低估总 tC 2.28×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RHODES 用 tC 公式（Eq.4）作为 MILP 的线性约束（Eq.10），不确定性经 `本库` 的 `碳强度（Carbon Intensity）`、`EPW/MPW/GPW`、`不确定性集` 条目所述方式编码；Julia + Gurobi 求解。验证用 Monte Carlo（每参数 2000 样本、normal/uniform/exponential/bernoulli/cauchy 分布、非负阈值拒绝重采样）算 p95 tC 与约束违反率。使用：设计师在早期设计阶段用 HILP profiling 数据（Rodinia 负载）评估候选 SoC 配置的 tC，权衡性能/功耗/面积/碳。开源：https://github.com/mariamelgamal/RHODES（仅 README，代码未公开）。相关工具：ACT [32]、FOCAL [21]、GreenChip [41]、CORDOBA [23]。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
