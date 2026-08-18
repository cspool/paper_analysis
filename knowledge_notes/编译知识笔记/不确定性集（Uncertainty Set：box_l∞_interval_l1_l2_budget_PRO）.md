## 不确定性集（Uncertainty Set：box/l∞/interval/l1/l2/budget/PRO）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
不确定性集 U 是鲁棒优化中界定"不确定参数所有可能实现"的集合，是 RO 的基础构件：把不确定参数 z 的可能取值约束在某个有界集合内，替代对其概率分布的完整知识。经典集合（论文 Eq.1）：box 集 U_l∞={z:‖z‖∞≤ρ}（区间 [-ρ,ρ] 内任意取值）、interval 集 U_interval={z:LB≤z≤UB}（用户显式给上下界）、l1 集 U_l1={z:‖z‖1≤ρ}、l2（欧几里得）集 U_l2={z:‖z‖2≤ρ}、budget 集 U_budget={z:‖z‖1≤ρ,‖z‖∞≤Γ}（利用中心极限定理，防止所有参数同时取极端值）。更信息化的概率鲁棒（PRO）集 U_PRO={z: −(1/n)Σ log P(z̃_i≥z_i) ≤ Γ}（Eq.2，Bertsimas & Boucher）：用可靠函数（log-survival，CDF 的补）捕捉包括重尾在内的整个分布信息，可区分均值方差相同但分布不同的数据；可由有序经验样本 y_{i,1}≤...≤y_{i,N} 用 P(z̃_i≥y_{i,j})≈(N+1−j)/N 构建，推广到纯经验数据场景。RO 的优势正在于 U 可只凭区间或少量样本构造，无需完整分布。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES 中不确定性集被编码进 MILP 的约束生成：
- DA 参数（FPW/GPW/MPW，CPU 核 c / CPU I/O die cIO / GPU SM g 各不同）→ U_interval，直接以 Table II 的 LB/UB（如 FPW_c 565.5–888.0 g CO2e/300mm 晶圆、GPW_c 65.7–247.1、MPW_c 318.1–388.8）写进 tC 约束（Eq.10 中 (FPW+GPW+MPW)⊙A 各项，∀FPW∈U^DA... 遍历集合所有成员）。
- DV 参数（Cop：CIuse×toperational×功耗×执行时间）→ U_PRO，用 CAISO 2022–2024 小时级 CIuse 样本（720 点/月）构建，Γ=1。
- 每个集合经鲁棒对等生成约束：U_PRO 的 robust counterpart（Eq.3）为每个数据样本加一条约束（nΓα+Σβ_i≤b；−log((N_i+1−j)/N_i)α+β_i≥x_i·y_{i,j}；α≥0，α/β 为对偶变量），用线性强对偶实现。
- WLP 案例用 U_budget 防止 EPW/GPW/MPW 在多个 SoC 组件上同时取最坏值导致过度保守。
设计者可通过调 Γ（robustness 参数）与选择集合类型自由控制保守程度与数据利用程度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RHODES 把不确定性集模块化（uFPW/uGPW/uMPW/uCop 等函数），每种集合一个函数生成对应 robust counterpart 约束；数学推导与代码见论文 GitHub（https://github.com/mariamelgamal/RHODES，当前仅 README）。使用：为每个不确定参数指定集合类型与参数（区间 LB/UB、Γ、CI 样本向量），框架自动生成向量与线性化约束供求解器处理。论文用 U_interval/U_l∞ + 调 Γ 探索鲁棒性-最优性权衡；DV 参数用 U_PRO 平均违反率最低。设计启示：随着碳数据积累，可从保守的 U_interval 升级到信息量更高的 U_PRO，减少过度保守。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
