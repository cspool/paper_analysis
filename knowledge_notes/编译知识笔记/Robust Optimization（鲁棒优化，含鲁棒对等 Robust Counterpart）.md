## Robust Optimization（鲁棒优化，含鲁棒对等 Robust Counterpart）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
鲁棒优化（RO）是一类在不确定数据下做决策的数学优化范式：不假设或依赖不确定参数的概率分布，而是把"不确定数值数据"与"已知确定的优化问题结构"分离，用确定性**不确定性集（uncertainty set）**界定不确定参数所有可能的实现范围，要求解对集合内所有实现都可行（robust feasible），从而把无限多个约束的问题通过对偶理论 reformulate 成等价的确定性问题——**鲁棒对等（robust counterpart）**——再交给常规优化求解器求解。经典文献：Bertsimas & Sim（Price of Robustness, 2004）、Bertsimas, Brown & Caramanis（Theory and Applications of RO, SIAM Review 2011）、Bertsimas & Den Hertog（Robust and Adaptive Optimization 教材）。与随机优化/贝叶斯优化/统计机器学习等方法不同，RO 不依赖分布估计，因此特别适合"数据稀少到无法可靠估计分布"的 Data Availability（DA）不确定性场景。RHODES（ISCA'26）首次把 RO 系统性地用到 CO2 感知硬件设计：碳参数同时含 DA 不确定性（EPW/MPW/GPW，每节点仅 1–4 个数据点）与 DV 不确定性（CIuse 有海量电网数据），RO 用区间/PRO 不确定性集同时处理两者，避免 KDE 等统计方法在稀疏数据下的错误结论（论文 Fig. 2 示例：稀疏数据下 p95 结论随数据增多反转）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RHODES（Julia 实现的优化框架）中 RO 的运转流程：
```
输入：设计约束(T_max,P_max,A_max,tC_max) + 碳参数不确定性集 + HILP workload profiling 数据
1. 分类不确定性：DA 参数(FPW/GPW/MPW/t_operational) → U_interval/U_l∞；
                   DV 参数(CI_use) → U_PRO(由 CAISO 小时级样本构建)
2. 线性化：CIfab·EPW → FPW，CIuse·t_operational → Cop（未知×未知=未知×常数，
           保持 tC 约束在不确定参数上线性）
3. 写出 robust counterpart：对每个不确定性集，用对偶理论把"对所有 z∈U 都成立"的
           约束折叠为确定性约束（U_PRO 每个数据样本加一条约束，Eq.3）
4. 转 MILP：二进制决策向量 c(CPU核), g(GPU SM/频率), x(GPU选择), w=c·x 线性化
5. 调求解器（Gurobi）解 MILP → 输出对所有 U 内实现均可行的 SoC 配置
```
鲁棒性参数 Γ 可扫描 0（无不确定性）→ 1（全集），在设计鲁棒性（更低碳）与目标性能（执行时间）之间连续权衡（论文 Fig. 12）。求解实例：45,088 个配置的单个 RO run <5 分钟、0 最优间隙（ARM 8 核 8 线程）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RHODES 在 Julia 中实现，模型 solver-agnostic（可换任意线性/混合整数求解器），本论文用 Gurobi；不确定性集模块化，用户按参数关系与数据可得性自选集合（box/interval/l1/l2/budget/PRO）。论文提供开源仓库 https://github.com/mariamelgamal/RHODES（截至检索仅有 README"updates coming soon!"，代码未公开）。使用：设计师给 SoC 配置空间（1–32 CPU 核、0–128 GPU SM×11 频率）、碳参数不确定性集与设计约束，RHODES 输出鲁棒最优配置与 tC/执行时间。应用效果：robust 设计比 nominal（不考虑不确定性）在 MC 评估下 tC 低 1.38–1.57×、tC 约束违反率从 49.85–54.4% 降到 ≤1.1%，tCDP 比 SOTA CORDOBA 好 1.3–3.17×。Web supplement：Bertsimas et al. https://doi.org/10.1137/080734510；Bertsimas & Den Hertog 教材（Dynamic Ideas LLC, 2022）；Bertsimas & Boucher《Probabilistic Robust Optimization》（U_PRO 集出处）。

涉及论文标题：
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
