## Schottky 接触电阻（Schottky Contact Resistance）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Schottky 接触电阻是金属-半导体接触处的势垒电阻。硅工艺用重掺杂有源区实现低阻欧姆接触，但 2D 材料掺杂困难（dopability 受限），其金属接触本质上呈 Schottky 势垒，接触电阻随偏压指数变化：R_sch ∝ exp(-V_gs) + exp(-V_gd)（TDMSim 式 6，V_gs/V_gd 为栅源/栅漏电压）。该电压相关电阻重塑 bitline 网络的 RC 特性——2D DRAM 阵列的位线充放电时间与能量与硅假设不同。TDMSim 在 TDM-Memory 的 bitline 模型中集成 Schottky 接触电阻：仿真时按阵列/cell 的 RC 抽象在当前电压下计算 R_sch 再并入位线模型（Schottky contact resistance model）。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级流程：TDM-Transistor 给出 I-V（含接触电阻效应）→ TDM-Memory 按式 6 计算当前偏压下的 R_sch → 并入 bitline RC 网络（与 gate 电容、drain 电容共同决定读/写/刷新的延迟与能量）→ array 级访问延迟/能量输出。接触电阻还参与 retention 梯度：边缘 cell 接触电阻变化加重 off-state 泄漏、缩短边缘 retention（径向梯度成因之一）。TDM-Memory 因此能定量评估"接触电阻对阵列不同位置读/写延迟的影响"（论文 IX-2 示例）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：器件/工艺上通过接触工程缓解（Au/Ti/Ni 复合电极、hBN 界面层、edge-contact、自对准工艺等，Table III 验证范围）；建模上以式 6 电压相关解析式集成进 CACTI 扩展。使用要点：2D 器件建模必须显式处理接触电阻（硅 CACTI 模型默认欧姆接触）；对 RC 网络、读/写窗口与刷新能量均有影响；评估中应做接触电阻敏感性分析。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
