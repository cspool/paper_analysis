## 负二项良率模型（Negative Binomial Yield Model，α、D0）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
负二项良率模型 Y = (1 + A·D0/α)^(−α)（Stapper，1973/1984）是 VLSI 最接近实际的 die 良率模型：A 为 die 面积，D0 为缺陷密度（平均缺陷数/cm²），α 为缺陷聚类参数。它由"泊松分布与 gamma 分布的缺陷密度复合"导出：α 越小缺陷越成簇（严重聚簇时坏 die 集中在少数区域），α→∞ 退化为泊松模型 Y=e^(−D0A)。α 可由 λ̄/(σ²−λ̄) 从缺陷数据估计，业界实测常见 2–3（Cunningham 1990 综述）。CAPA 论文的缺省参数：成熟制程 α=10、D0=0.15 cm⁻²（[13]），分析用 D0=0.1 cm⁻²；面积 400→800mm² 时良率从约 56% 降到 32%。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CAPA 用两种方式：(1) 无 binning 基线：直接用 Eqn. 1 得 Y_die，代入 C_die=CPW/(N_die×Y_die)；(2) binning 良率：Eqn. 7 给出"die 恰有 d 个缺陷"的负二项概率 P_defect(d)（β=D0·A/α，d=0 时退化为 Eqn. 1），再与模块级缺陷分布（Eqn. 8-10）卷积求和得到 binning 良率（Eqn. 9/11）。interposer/EMIB 用不同参数（D0=0.06 cm⁻²、α=6）。敏感性分析：α 扫 ±3 对整包碳影响 <0.075%（可忽略）；D0 扫 ±0.03 时，逻辑 die 主导的 56 核 SPR 整包碳最大波动 7.12%，H100 仅 2.24%（H100 碳由 HBM 主导、逻辑 die 只占 13%）——即 D0 的碳传播路径是"D0→良率→逻辑 die 碳→整包碳"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
良率数据是 fab 机密，架构研究通常从公开报道取值（论文引用 TSMC N5 良率报道 [13]、Chiplet Actuary [25]）；实现上公式只需 Gamma 函数（Eqn. 7）与 Stirling 数（Eqn. 8）两个特殊函数，CAPA 用 Python（scipy）实现。使用要点：α 对整包碳几乎无影响，D0 是高敏感参数——碳模型的验证精度本质上受限于 D0 的取值（论文 TPUv4 用 D0=0.1 得 91.9 kgCO2eq、0.4% 误差，换 0.09 则 1.4% 误差）。

RHODES 的 Murphy 良率视角（ISCA'26）：RHODES 用更简的 Murphy yield model Y=e^(−A·D0)（Murphy, 1964）代替负二项（负二项在 α→∞ 时退化到泊松 Y=e^(−D0A)，Murphy 即此单参数形式），针对成熟节点（7nm EUV 2018 量产、12nm 2013 量产）确定性建模：D0=0.1–0.15 defects/cm²，良率不作为显式不确定性参数（其面积依赖已被 tC 公式 A/Y 项内建），但在 §V-C 评估架构冗余 SM/核（redundancy）对良率与碳效率的影响：n_SMsTotal 个 SM 中至少 n_SMsNeed 个功能正常的概率 = Σ_{k=n_SMsNeed}^{n_SMsTotal} C(n_SMsTotal,k)·(e^(−A_SM·D0))^k·(1−e^(−A_SM·D0))^(n_SMsTotal−k)。2–4 个冗余 SM 可显著提升 tCDP（2 个冗余时最高 2.98×），过度冗余则内含碳开销超过良率收益（收益递减）。

涉及论文标题：
- CAPA: Manufacturing Carbon Estimation for Advanced-Packaged Architectures
- RHODES: Robust Optimization for Uncertainty-Aware Design of CO2-Efficient Computing Systems
