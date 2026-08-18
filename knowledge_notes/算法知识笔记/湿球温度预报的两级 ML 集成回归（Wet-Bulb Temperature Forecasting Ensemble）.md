## 湿球温度预报的两级 ML 集成回归（Wet-Bulb Temperature Forecasting Ensemble）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 湿球温度（WBT）是蒸发冷却能力的极限温度：冷却塔通过蒸发水分把热量排入环境，蒸发潜力越高（WBT 越低）冷却能力越强；气候模拟（CMIP6）直接提供干球温度（DBT）与相对湿度（RH），但因为数据时间错位、网格粗，无法用标准公式直接算出 WBT。Prometheus（Google, ISCA 2026）用"两级 ML 集成回归"解决：第一阶段两个基回归器——随机森林（RF：100 棵树、max_depth=5、每分裂节点最少 2 样本、输出为树均值，防过拟合的浅树）与支持向量机（SVM：RBF 核，把日 min/mean/max DBT 与 RH 共 6 个特征映射到高维空间使非线性关系可分）——各自独立预测日最大 WBT；第二阶段神经网络（NN：两个隐层 16/8 神经元、ReLU 激活、L2 正则 0.5、输出层单神经元无激活）以两个基模型预测 + 原始 6 特征为输入，输出单一稳健的 WBT 预报。核心动机：多种互补模型集成降低方差、提高极端值（99.5 百分位）预报精度——Table I 显示集成 RMSE 0.67°C vs 最佳单一 baseline 1.71°C（-43%）、99.5 百分位正误差 3.5 vs 9.7（-60%+），因为数据中心设计恰恰依赖分布尾部的极端温度。
- 该术语从算法pipeline角度拆解：这是"特征→两个独立基学习器→元学习器融合→偏差校正→极值分布拟合"的完整预报 pipeline。伪代码：
  ```
  # 输入：0.25° 网格上 CMIP6 投影的日最低/平均/最高 DBT 与 RH（X ∈ R^(6)）
  y_rf  = RF(n_estimators=100, max_depth=5, min_samples_split=2).predict(X)   # 每棵树叶子均值，再平均
  y_svm = SVR(kernel='rbf').predict(X)                                        # K(x,xi)=exp(-γ||x-xi||²)
  y_nn  = NN(hidden=[16,8], activation=ReLU, L2=0.5).predict(concat(X, y_rf, y_svm))  # 单输出无激活
  y     = y_nn - bias(site)      # 地理偏差校正（5 年历史均值差）
  # 之后：Gumbel 拟合 → N 年回返温度（见独立条目）
  ```
  张量计算：RF 沿特征阈值分裂、输出 = 落入叶子的训练样本均值；SVM 核函数 K(x,x_i)=exp(-γ‖x-x_i‖²)；NN 前向 h1=ReLU(W1x+b1)（16 维）→ h2=ReLU(W2h1+b2)（8 维）→ ŷ=w3·h2+b3（标量）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：SVM/RF/NN 均为标准监督回归模型，可用 scikit-learn（SVR/RandomForestRegressor）与任意深度学习框架实现；输入来自公开 CMIP6 数据（经 Google Data Commons 访问，0.25° 网格、日 DBT/RH 到 2100 年）；6 个 CMIP6 模型构成覆盖高/中/低平衡气候敏感度（ECS>4K / 2.87-4K / <2.87K）的集合以包含不同未来情景。训练每站点仅需数小时，推理成本可忽略（决策在年度/十年尺度，不在关键路径）。
  - 使用：对每个数据中心站点，用 25 年历史观测 + 20 年 CMIP6 前向投影拟合模型，输出该站点未来日最大 WBT 分布，再交给 Gumbel 拟合得到 50 年回返温度，驱动冷却容量设计。论文未提供代码仓库链接，联网检索（2026-08）未发现公开实现，无法确认开源。

涉及论文标题：
- Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure
