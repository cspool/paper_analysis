## Gumbel 极值分布与 N 年回返温度（Gumbel Distribution and N-Year Return Period Temperature）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Gumbel 分布（I 型极值分布，也称 Fisher–Tippett I 型）是极值理论中用于建模"每年最大温度/降水"等块最大值的概率分布，适合数据中心"设计温度"这种低频极端事件风险量化。其累积分布 F(x;µ,β)=exp[-exp(-(x-µ)/β)]，位置参数 µ 与尺度参数 β 由均值 T_max-mean=µ+βγ（γ=欧拉常数 0.5772）与标准差 T_max-std=πβ/√6 反推。N 年回返温度 T_N 指"平均每 N 年才被超过一次的温度"，对应 Gumbel 分布的 1-1/N 百分位：T_N = T_max-mean - (√6/π)[0.5772 + ln(ln(N/(N-1)))]·T_max-std。ASHRAE 只用量化前 30 年气象站历史数据拟合，忽略气候变暖；Prometheus 用"25 年历史观测 + 20 年 CMIP6 多模型投影"的混合数据拟合 µ/β，把未来气候情景的方差嵌入分布参数，从而把 London 2022 的 40.2°C 从 ASHRAE 的"1-in-200 年"修正为"1-in-50 年"，DBT/WBT 50 年回返温度相对 ASHRAE 平均高 4.4°C/1.4°C。
- 该术语从算法pipeline角度拆解：它是 ML 预报输出的概率后处理层，把"逐日 WBT 预报序列"转成"可决策的年超温概率"。pipeline 伪代码：
  ```
  # 输入：站点 s 的历史观测 + CMIP6 投影的年最大 WBT 序列 {T_max_year}
  T_mean, T_std = mean({T_max_year}), std({T_max_year})
  beta = T_std * sqrt(6)/pi                 # 尺度参数
  mu   = T_mean - beta * 0.5772             # 位置参数（γ=欧拉常数）
  # N 年回返温度（对应 1-1/N 百分位）：
  T_N  = T_mean - (sqrt(6)/pi) * (0.5772 + ln(ln(N/(N-1)))) * T_std
  # 年超温概率：Pr[annual max T > T_design] = 1 - F(T_design; mu, beta) = exp(-exp(-(T_design-mu)/beta)) 取补
  # 例：London 2044, SSP5-8.5: T_50(DBT)=41.2°C vs ASHRAE 2021: 37.7°C
  ```
  该概率直接喂给升级判据式 5：Pr[Annual max T > 冷却设计温度]_Gumbel > 2% 时触发冷却升级（对应 1-in-50 年事件风险容限）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：Gumbel 参数可用极大似然估计或矩估计（上式即矩估计）；Python 中 scipy.stats.gumbel_r / gumbel_l 提供现成分布与拟合函数。论文未提供代码，联网检索未发现公开实现，无法确认开源。
  - 使用：Gumbel 分布是工程上（水文、建筑、气候）估计设计回返期的标准工具（IEEE/ASHRAE 设计工况、大坝防洪等均用类似极值方法）。在 Prometheus 中它把"气候模型投影的不确定性"转化为"冷却容量该配多少"的定量依据，也是与 ASHRAE 后向方法对比的桥梁（同样输出 50 年回返温度，但输入数据不同）。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure
