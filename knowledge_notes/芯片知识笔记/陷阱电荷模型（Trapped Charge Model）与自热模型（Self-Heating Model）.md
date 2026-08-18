## 陷阱电荷模型（Trapped Charge Model）与自热模型（Self-Heating Model）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这两个是 TDMSim 对基线 2D 晶体管模型（2DFETs/BSIM-CMG）的两个器件级扩展，使解析模型贴近可制造 2D 器件。(1) Trapped Charge Model：2D 器件制造常用 e-beam 蒸发 ultrathin 金属 seed layer 以促进 HfO2 的 ALD，该工艺界面易产生结构/化学缺陷 → 显著陷阱电荷，扰动电容、阈值电压与亚阈值摆幅。模型在电荷平衡方程加入陷阱电荷密度 Q_t：C_ox(V_as - V_fb - φ_s) = Q_m + Q_D + Q_t，其中 Q_t = q·Σ_i D_trap,i/(1 + exp((V_ch - φ_s + E_it,i/q)/V_t))（式 1-2，D_trap,i/E_it,i 为第 i 个陷阱能级的陷阱密度与相对导带底的陷阱能级）。(2) Self-Heating Model：2D 器件高电流密度 + 高界面热阻使自热显著（硅模型的热假设不适用）。用 BSIM-CMG 的等效热网络建模：R_TH = R_B + R_i + R_Si（2D 沟道与底绝缘体间的热边界电阻、绝缘体扩散电阻、衬底扩散电阻），因 R_B∝1/W、R_i~1/W、R_Si∝1/√W 合并为 R_TH = R_TH0/W + R_TH1/√W（式 3，W 为有效器件宽度）。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级流程：TDM-Transistor 输入 process/temperature/voltage 参数 → 陷阱电荷模型把实测陷阱密度（从 C-V/I-V 提取）映射到 Vth/亚阈值摆幅/电容的偏移 → 自热模型把功耗折算为沟道温升（影响 off-current 与 retention）→ 输出晶体管 I-V/电容/泄漏 → TDM-Memory 用其合成 cell/array。验证：模型复现流片晶体管的 Id-Vg/Id-Vd（Fig.6，on-state >200 A/m、off-state <10^-12 A/μm 受仪器分辨率限制、实测 off-current 约 10^-17 A/μm @330K），优于基线模型。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 2DFETs（BSIM-CMG 框架）解析模型中加入 Q_t 项与 R_TH 热网络；陷阱参数从实测 C-V/I-V 提取标定，自热系数 R_TH0/R_TH1 经验拟合。使用要点：陷阱电荷与自热是 2D 器件"真实可制造性"的代表非理想性，架构级仿真若忽略会高估 retention/低估延迟与泄漏；TDMSim 的自动化标定流程通过关键电参数（on-current、亚阈值摆幅、Schottky 电阻）覆盖不同 2D 材料/结构（Table III 已验证 MoS2/WS2 等 8 类器件），无需为新材料重写模型。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
