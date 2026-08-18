## 制造碳（Embodied / Manufacturing Carbon，Scope 1/2/3、CPW、LCA、PCF）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
制造碳（embodied carbon / manufacturing carbon）指计算硬件在生产、组装、部署全流程产生的温室气体排放，以 kgCO2eq 计，区别于使用期能耗产生的运营碳（operational carbon）。核算体系上遵循 GHG Protocol 把排放分为 Scope 1（fab 直接排放，如工艺气体）、Scope 2（外购电力，取决于 fab 所在地电网碳强度 CI_fab）、Scope 3（上游材料、化学品、设备等供应链排放）。CAPA 的核心把每片晶圆的制造碳 CPW 分解为 CPW = S1PW + S2PW + S3PW，其中 S2PW = EPW（每晶圆能耗）× CI_fab（Eqn. 4-5）。LCA（生命周期评估）是这套核算的方法论；PCF（产品碳足迹）是厂商披露格式（如 NVIDIA HGX H100 PCF 报告）。超大规模云厂商 2024-25 报告中 embodied carbon 占其总排放的 54–91%（Microsoft/Meta/Google）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
芯片设计层面的关键是把"晶圆级碳"折算为"单个功能裸片碳"：fab 处理的整片晶圆碳固定，需要按能切出多少功能 die 来分摊——C_die = CPW / (N_die × Y_die)（Eqn. 12）。N_die 由 gross die per wafer（Eqn. 6）给出：圆晶圆边缘 3mm 排除 + 60µm 划片道让大 die 在晶圆上排布效率更低、废硅更多。因此良率越低、die 越大，单 die 碳越高。具体例子：800mm² die、D0=0.1/cm² 时良率约 47%，实际单 die 碳 = CPW/(68×47%) ≈ 0.03 CPW；而 ACT 用 100% 良率 100mm² 小 die 的碳/面积 CPA 线性外推只得到 ≈0.006 CPW——这就是 ACT 低估大 die 55% 的根源。流程链：设计参数（面积、节点、D0、α、fab 位置）→ 良率模型 → 单 die 碳 → 沿封装树逐节点聚合 → 整包制造碳。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
数据源是 imec.netzero（N65–N2 各节点的 CPW 分解，EUV 自 N7 引入）、TechInsights（HBM 碳）与厂商 PCF 报告；CAPA 开源实现中这些作为查表参数，用户只输入面积、节点、binning 参数等高层信息。业界实践：imec 于 2023-11 公开其"虚拟晶圆厂"web 应用（Scope 1+2 公开），数据显示 3nm 逻辑制程中光刻+刻蚀占 Scope 1+2 排放约 45%，且 IC 制造 CO2 排放预计十年内翻四倍——这正是 embodied carbon 建模的工具生态背景。

涉及论文标题：
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
