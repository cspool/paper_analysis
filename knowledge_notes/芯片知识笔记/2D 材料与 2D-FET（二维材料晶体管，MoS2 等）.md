## 2D 材料与 2D-FET（二维材料晶体管，MoS2 等）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 2D 材料指原子级厚度（单层数纳米以下）的层状材料，如过渡金属硫族化合物 MoS2、WS2、石墨烯、h-BN 等。用 2D 材料作沟道的场效应晶体管（2D-FET）具有两个对存储设计至关重要的物理特性：(1) 原子级薄沟道带来优异的静电控制（亚阈值摆幅接近 60mV/dec 理论极限）与超低泄漏（TDMSim 流片实测 off-current 约 10^-17 A/μm @330K，峰值 10^-15 A/μm，远低于硅）——这让 DRAM 存储电容的电荷泄漏极慢、retention 时间大幅延长；(2) 高集成密度（更小器件尺寸、更短 wordline/bitline 布线）降低寄生电容与互连延迟，从而降低动态能量。此外 2D 材料可实现更小存储电容（低泄漏允许）进一步提升密度与动态能效。业界进展：IMEC/IRDS 已给出 2D 集成路线图，TSMC/Intel/IBM 报道了 2D 原型（论文引 [12]-[17]）；Fudan 团队报道了基于 MoS2 的 32-bit RISC-V 处理器（约 5900 个晶体管，Nature 2025）。TDMSim 用 CVD 单层 MoS2/sapphire 衬底、30nm 沟道、130nm CGP、top-gate、HfO2 高 k 栅介质、Au 电极的晶体管做模型标定。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 TDMSim 的 2D-material DRAM cache 中，MoS2 2D-FET 作为 1T1C/3T0C 单元的访问/存储晶体管与外围电路的开关器件：原子级薄体使 gate 电容受量子电容主导（见量子电容条目）、SOI 式无结电容（见 drain capacitance 模型）、Schottky 接触电阻重塑 bitline RC（见 Schottky 条目）；低泄漏让 1T1C 可用更小存储电容并保持 0.5s 级 retention（硅仅 64ms）、3T0C 的 gate 寄生电容存电荷方案才可行（retention 0.1s）。芯片级流程：输入 2D 器件参数（I-V、电容、接触电阻）→ TDM-Memory 在 32MB SRAM 面积预算下合成 cell/array → 输出访问延迟/能量/面积 → gem5 系统级评估。2D-1T1C 在等面积下达到 512MB（约 16× SRAM 容量），静态功率在 512MB 时仅约 SRAM 87%、同容量 Silicon-1T1C 的 30.6%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CVD/机械剥离生长 2D 材料 → e-beam 光刻/ALD/蒸发制造晶体管（TDMSim 流片用 e-beam 蒸镀 ultrathin 金属 seed layer 促进 HfO2 ALD、电子束光刻 Au 源漏、ALD HfO2 栅介质、e-beam 蒸发 top Au gate）；建模用解析 compact model（2DFETs/BSIM-CMG 框架 + trapped charge/self-heating 扩展）。使用：作为新兴存储器件研究中的器件级输入，经跨层仿真（TDMSim：TDM-Transistor→TDM-Memory→gem5）传播到架构级决策；TDMSim 已验证器件范围（Table III）：MoS2/WS2、top/back/dual-gate、top/edge-contact、30nm-5µm 沟道、Au/Ti/Ni 复合电极、hBN 界面层、自对准工艺等。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
