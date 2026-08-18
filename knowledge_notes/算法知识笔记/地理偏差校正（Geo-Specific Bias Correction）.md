## 地理偏差校正（Geo-Specific Bias Correction）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 偏差校正是气候降尺度/预报的后处理技术：全球/区域气候模型（CMIP6）输出在特定站点存在系统性偏差（模型网格点与站点海拔、局地微气候不符），直接把模型输出当站点真值会导致持续偏移。Prometheus 采用站点特异性偏差校正：把某站点模型预报与同站 5 年历史观测对比，计算两者均值差 bias = mean(y_nn_hist - y_obs)，再从集成输出中减去该偏差 y_corr = y_nn - bias，从而消除系统性误差、使站点级预报与历史记录一致。这是大空间数据集（如 NEX-GDDP-CMIP6 降尺度数据）中"bias-corrected"产品的标准做法（论文引用 [7]）。
- 该术语从算法pipeline角度拆解：它是 ML 集成之后的轻量校准层，输入是"集成输出的日最大 WBT 序列"，输出是"与该站点气候一致的校正序列"，之后才进入 Gumbel 拟合。伪代码：
  ```
  # 站点 s：用最近 5 年历史观测校准
  bias_s = mean_over_years( ensemble_forecast_s[t] - observed_wbt_s[t] )
  y_corrected_s = y_nn_s - bias_s        # 逐站点偏差可正可负（站点高于/低于模型网格）
  # 校正后序列进入 Gumbel 拟合（µ, β），再算 T_50
  ```
  作用：0.25° 网格（约 27km）仍比单站点粗，偏差校正把"区域平均预报"归一到"站点实际气候"，是支撑"站点特异性概率风险评估"的关键一环（论文称减少系统性误差）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：本质是简单均值偏移校正（delta method 的一种简化）；更一般的气候学中还有分位数映射（quantile mapping，校正整个分布而非仅均值）。论文未提供代码，联网检索未发现公开实现，无法确认开源。
  - 使用：Prometheus 对 30 个生产数据中心逐站点应用该校正，使预报与各站历史气候一致；配合 ML 集成与 Gumbel 拟合，产出站点级 50 年回返温度（Table II 中 Dublin/London/Phoenix/Council Bluffs/Dalles 的具体数值即校正后结果）。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure
