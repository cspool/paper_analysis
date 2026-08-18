## tCDP（总碳延迟积，Total Carbon-Delay Product）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
tCDP（total carbon-delay product）是衡量计算系统碳效率的指标，定义为总碳足迹 tC 与执行时间（延迟）T 的乘积，由 Elgamal et al. 在 CORDOBA（HPCA 2025）[23] 提出：tCDP = tC × T。它同时权衡制造碳、运营碳与能效——低碳但极慢的系统与快但高碳的系统都有较高 tCDP，tCDP 越低表示"每单位任务完成时间付出的碳"越少。相比单独看 tC 或能效，tCDP 是综合碳效率的设计目标。RHODES 把 tCDP 作为第三个优化目标：由于 tC×T 在决策变量上非线性，框架用"递增 tC 阈值序列上逐个最小化执行时间"扫出 tC-T Pareto 前沿，再找 tCDP 最优设计，保持 MILP 线性。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 中 tCDP 优化流程：对 tC 阈值序列（如 1–100 kg CO2e）逐点解"min T s.t. 约束(6)-(9)+tC≤阈值"的鲁棒 MILP → 得到 Pareto 点集 → 计算每点 tCDP=tC×T → 取最小者为鲁棒最优设计。效果（论文 §V-C）：RHODES 设计的 tCDP 比 nominal 设计与 SOTA CORDOBA 好 1.3–3.17×；相比 CORDOBA 只扫 operational time、其余参数取标称值并输出多个候选让设计者手动挑选，RHODES 把多参数不确定性编码进优化、自动选单一设计（如 heartwall 选 c1g16，碳效率优于 CORDOBA 全部候选 c1g2/c1g4/c1g10/c1g14/c1g19），并把 nominal-to-real gap 从 1.23–3.49× 缩到 1.01–1.12×。附加实验：GPW abatement 不确定性（95% vs 近 0%）使鲁棒设计变化但 tCDP 仅差 1.02×（运营碳主导时制造不确定性仍可移动最优）；GPU 主导 SoC 加 2 个冗余 SM 使 tCDP 提升 1.01–2.98×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RHODES（Julia+Gurobi）用 Pareto 追踪法线性化 tCDP；对比框架 CORDOBA [23]（HPCA 2025）用 Lagrange multiplier 处理未知 CIuse + 启发式处理不确定 operational time，离散扫描 operational time 输出候选设计。使用：设计师把 tCDP 作为目标，指定 tC 阈值扫描范围，得到碳效率最优且对不确定性鲁棒的 SoC 配置（CPU 核 + GPU SM/频率）。与 PPA 目标的权衡（Fig. 8 三种目标：min tC、min T、min tCDP）由同一框架支撑。开源：https://github.com/mariamelgamal/RHODES（仅 README）。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
