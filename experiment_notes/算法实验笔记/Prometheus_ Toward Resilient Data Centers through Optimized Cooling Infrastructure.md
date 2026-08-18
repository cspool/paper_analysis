## Prometheus: Toward Resilient Data Centers through Optimized Cooling Infrastructure

- 属于算法pipeline的实现是什么？实验比较什么？
  - Prometheus 的核心算法实现是"多阶段 ML 集成回归"用于预报湿球温度（WBT）——CMIP6 物理气候模拟不直接提供 WBT，但蒸发冷却塔的冷却能力以 WBT 为极限，因此必须用 ML 从模拟数据推断。pipeline：输入 0.25° 网格上 CMIP6 投影的日最低/平均/最高干球温度（DBT）与相对湿度（RH）共 6 个特征；第一阶段两个基回归器各自独立预测每日最大 WBT——随机森林（RF，100 棵树、max_depth=5、每分裂节点最少 2 样本、输出为树平均）与支持向量机（SVM，RBF 核把输入映射到高维空间）；第二阶段神经网络（NN，两层隐层 16/8 神经元、ReLU、L2 正则 0.5、输出层单神经元无激活）以两个基模型预测 + 原始特征为输入，输出单一稳健 WBT 预报。其后做站点特异性偏差校正（用该站点 5 年历史数据算均值差并从输出中减去），最后用 Gumbel 分布（式 2，位置 µ/尺度 β）拟合"25 年历史观测 + 20 年 CMIP6 多模型投影"混合数据，由式 3 由均值 T_max-mean=µ+βγ 与标准差 T_max-std=πβ/√6 计算 N 年一遇回返温度 T_N（对应 1-1/N 百分位）。
  - 实验比较：①ML 精度（Table I）：比较 baseline1（解析公式直接从 DBT/RH 统计量算 WBT）、单独 SVM、单独 RF、Prometheus 集成——RMSE 1.71/0.70/0.71/0.67°C（集成降 43%）；99.5 百分位正误差 9.7/5.0/7.0/3.5（降 60%+，极端值预报最关键的场景），99.5 百分位净误差 -6.4/-2.7/-2.8/-2.4；②与 ASHRAE 历史对比（§VI-B 回测）：对历史年份用 25 年历史观测 + 20 年 CMIP6 投影"回测"1-in-50 年温度并对比官方 ASHRAE 值——DBT 差 -2.8~19.7°C（均值 4.4°C）、WBT 差 -4.0~12.1°C（均值 1.4°C），单个站点 WBT 差可超 3°C；③固定裕量对比（Fig.9）：3°C WBT 裕量中位低估 0.69°C、25% 站点低估超 1.4°C；6°C DBT 裕量对一半机群欠配、另一半过配；④未来排放情景（§VI-C）：SSP2-4.5/SSP5-8.5 下 2044 年 50 年回返温度 WBT 高 2.7~6.8°C、DBT 高 2.0~10.7°C；⑤生产部署（§VII）：30 个数据中心，冷却容量需平均增 11%、最挑战站点增 39%（DBT）/48%（WBT），12% 站点当前即有 >2% 年概率超设计温度。
- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/推理硬件（Google 内部生产环境）。计算需求（§V）：使用公开 CMIP6 数据，避免物理气候模拟的巨大算力；ML 集成训练每站点仅需数小时，推理成本可忽略且不在关键路径上（决策在年度/十年尺度）。数据经 Google Data Commons（https://datacommons.org）访问。
- 模型是什么。数据集和bench分别是什么。
  - 模型：两级集成——SVM（RBF 核）+ RF（100 棵树、max_depth=5、min_samples_split=2）→ NN（2 隐层 16/8、ReLU、L2=0.5、输出单神经元无激活）。输入 6 特征（DBT/RH 的日 min/mean/max），输出日最大 WBT。基线模型：ASHRAE 后向统计、GSTR 缩放法（式 1）、解析 WBT 公式（baseline1-3：max DBT + 日 min/mean/max RH）。
  - 数据集：CMIP6 六模型集成（覆盖低/中/高平衡气候敏感度 ECS：<2.87K / 2.87–4K / >4K，如 NEX-GDDP-CMIP6 降尺度），0.25° 网格、每日 DBT/RH 最小/最大值到 2100 年；每站点 25 年历史观测（1965-1990 或 1999-2024 等）+ 20 年 CMIP6 前向投影。Bench：30 个生产数据中心（北美/欧洲/中东/南美/亚太），公开 5 个站点：Dublin、London、Phoenix（DBT 敏感）、Council Bluffs、Dalles（WBT 敏感）；回测用 London St. James's Park（WMO:037720, 1994-2019）。
- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供代码仓库链接（Google 内部框架），联网搜索未发现公开仓库，无法确认开源；依赖公开数据源 Google Data Commons（CMIP6 数据）与 NOAA GFS（两周运营预报）。
  - 算法 pipeline 例子（伪代码，站点 s 的年最大 WBT 预报 → 50 年回返温度）：
    ```
    # 1. 输入：0.25° 网格上 CMIP6 投影的日最低/平均/最高 DBT 与 RH（X ∈ R^(6)）
    # 2. 第一阶段基回归器（各自独立预测日最大 WBT）
    y_rf  = RF(n_estimators=100, max_depth=5, min_samples_split=2).predict(X)  # 100 棵树叶子均值平均
    y_svm = SVR(kernel='rbf', gamma=γ).predict(X)                             # K(x,xi)=exp(-γ||x-xi||²)
    # 3. 第二阶段元回归器
    y_nn = NN(hidden=[16,8], ReLU, L2=0.5).predict(concat(X, y_rf, y_svm))     # 输出层单神经元无激活
    # 4. 站点偏差校正（5 年历史）
    bias = mean(y_nn_hist - y_obs);  y_corr = y_nn - bias
    # 5. Gumbel 拟合（式 2/3）：β = T_max-std·√6/π，µ = T_max-mean - β·γ（γ=Euler 常数 0.5772）
    # 6. N 年回返温度：T_N = T_max-mean - (√6/π)[0.5772 + ln(ln(N/(N-1)))]·T_max-std
    #    → 伦敦 2044: T_50 = 41.2°C (SSP5-8.5) vs ASHRAE 2021: 37.7°C
    ```
    张量计算细节：RF 每棵树沿特征阈值分裂、输出 = 落入叶子的训练样本均值（平均 100 棵树）；SVM RBF 核 K(x,x_i)=exp(-γ||x-x_i||²) 把 6 维输入映射到高维使非线性可分；NN 前向 h1=ReLU(W1x+b1)（16 维）、h2=ReLU(W2h1+b2)（8 维）、ŷ=w3·h2+b3（标量 WBT），L2 正则 0.5 防过拟合；式 1 缩放法 T50_3C = [ln(50)-ln(9.4)]/[ln(9.4)-ln(3.7)]·(T50_1C-T10_1C) + T50_1C 作为对比基线。
