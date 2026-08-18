## ACmin（最小攻击行激活计数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ACmin 是 RowPress（ISCA 2023）引入、衡量 DRAM 芯片读干扰脆弱性的关键指标：在给定访问模式下诱发 victim 行**至少一个 bitflip** 所需的最少 aggressor 行激活次数（双面模式中两 aggressor 各算一次）；ACmin 越低 = 越脆弱。DejaVu（ISCA 2026）把它作为"写入历史影响"的标准测量对象：在 Baseline/OverWrite/SameWrite 三种初始化下各测 50 次重复并取最小 ACmin 做归一化对比。测量方法：二分法迭代逼近——每轮按初始化流程重写 victim/aggressor 行，以候选激活次数 hammer 后检查 bitflip 决定上下界，直至相邻两次迭代差 <10 收敛；每轮严格 <64ms（DDR4 刷新窗口）且期间不发 auto-refresh，排除 retention bitflip 干扰。Web 来源：RowPress（ISCA 2023，github.com/CMU-SAFARI/RowPress）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- ACmin 是"行级读干扰阈值"（文献亦写作 N_RH）的**最坏情况估计**：50 次重复取最小以捕捉跨行的器件变异性（DejaVu 观察到离群行变化最大 46.1%）。芯片设计中的意义：(i) 作为 RowHammer/RowPress 缓解阈值（TRR、PRAC 的 N_BO、控制器侧计数/概率刷新）设定的实验依据；(ii) 作为数据模式、温度（50/80 °C）、时序（tAggON=36ns 为 RowHammer、更大为 RowPress）敏感性研究的一维标量投影；(iii) DejaVu 表明 ACmin 依赖 victim 行写入历史——"如何初始化 victim 行"决定测量代表真实系统最坏情况（OverWrite）还是消除干扰的干净基线（SameWrite），因此测量协议本身是芯片测试方法学的一部分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：二分法 + 每轮重初始化 + 严格时间预算（如上）。典型使用流程（DejaVu Listing 1 伪代码）：init_rows(R, aggr_data, victim_data, case) 按 case∈{Baseline, OverWrite, SameWrite} 写 victim 行 → 两 aggressor 行交替 ACT（tAggON 控制 RowHammer/RowPress）→ 读回 victim 行比对 bitflip → 二分法收敛输出 ACmin。用于：芯片/模块横向对比（112 芯片、1792 行）、缓解技术阈值标定、写历史效应（DejaVu）的定量表征（归一化 ACmin 比值的 box/whisker 图）。

涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
