## Monte Carlo Simulation（蒙特卡洛模拟，TCO 容量规划方法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Monte Carlo Simulation 是通过大量随机抽样刻画不确定性的数值方法：把输入建模为随机变量，重复模拟得到输出分布而非点估计。本论文（Rearchitecting the Datacenter Lifecycle for AI）用 8 个随机变量（表 IV）驱动 TCO 生命周期模拟——workload 年增长（Log-normal µ=log(1.05), σ=0.05，与模型增长相关 ρ=0.4）、模型规模增长（Log-normal，按历史 P50 拟合、±2σ 截断）、GPU 性能/Watt 提升（Normal，与价格改善相关 ρ=−0.5）、GPU 代际价格（Triangular −15%/0%/+20%）、发布间隔（{1,1.5,2} 年）、电价（Log-normal σ=15%）、冷却效率 PUE（Normal σ=0.05）、server 生命周期（{4,5,6} 年）。从多元正态（协方差矩阵 Σ 编码经验相关性）抽样，经逆 CDF 变换映射到各边缘分布；每 trial 确定性地计算容量规划、采购/退役与年化 CapEx/OpEx，得到一条 TCO；重复 10,000 次（验证 20,000 次均值与 95% CI 变化 <1%）得到 TCO 分布。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
蒙特卡洛驱动的刷新策略评估流程：
```
for trial = 1..10000:
    采样 8 个随机变量（工作负载增长、模型增长、GPU 效率/价格、发布间隔、电价、PUE、寿命）
    for policy ∈ 枚举策略（各代生命周期 0–10 年，可跳过，多代共置）:
        for quarter t = 1..60:                    # 15 年
            goodput = roofline+ SLO(400/100ms)    # 性能
            需求 RPS(t) 按采样增长率累加
            采购/退役 → CapEx/OpEx → TCO(t)
        TCO[policy] = Σ TCO(t)
输出: 每策略的期望 TCO、方差、95% CI、策略间胜出概率、
     Sobol-style 一阶敏感性（回归分解）
收敛校验: 运行均值稳定(<1%/最后2000样本)、5/95 分位稳定、bootstrap CI
```
对每种刷新策略（如 aggressive vs delayed）都跑全流程，得到"该策略在不确定性下的 TCO 分布"——图 11a 显示通用数据中心 5 年刷新仍是多数策略中最优，图 11b 显示 AI 场景大量替代策略可降 15–20%，图 12 逐代展示 0–10 年寿命对 TCO 的影响（结论：V100→A100 该换，B100/B200 该跳，6 年以上寿命仍划算）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：任意数值库（numpy/scipy）按指定分布抽样 + 确定性仿真循环；论文的实现即开源 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）的 Monte Carlo 引擎——CLI `dc-tco monte-carlo --config configs/default.yaml --trials 10000 --output results/`，YAML 中可配置随机变量的分布、协方差结构与 trial 数。除基准趋势外，论文还用结构化突变场景（需求冲击 α=3 → TCO 降 31%、模型收缩 β=0.8 → 43%、硬件能力跳变 γ=3 → 38%、价格冲击 δ=0.6 → 36%）测试框架的鲁棒性，说明蒙特卡洛+场景枚举使固定刷新启发式在突变下失效的问题被显式量化。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
