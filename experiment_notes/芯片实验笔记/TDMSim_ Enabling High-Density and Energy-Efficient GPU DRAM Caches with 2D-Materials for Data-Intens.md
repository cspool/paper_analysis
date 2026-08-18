## TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications

- 属于芯片设计的实现是什么？实验比较什么？
  - 实现是 2D-material DRAM cache 芯片结构设计与建模：以 2D 材料（CVD 单层 MoS2/sapphire）晶体管（30nm 沟道、130nm CGP、top-gate、HfO2 高 k 栅介质、Au 电极）构建 1T1C 与 3T0C DRAM cell 及存储阵列，作为 GPU LLC。芯片设计空间探索在 32MB 6T-SRAM LLC 的版图面积预算约束下进行（外围电路与互连统一用 30nm 硅工艺），评估 Silicon-1T1C、2D-1T1C、2D-3T0C 三种结构在容量增长下的访问延迟、动态能量与静态功率（含 refresh power）；刷新策略遵循 JEDEC DDR4 标准，Silicon-1T1C 每 64ms 刷新，2D-1T1C 每 0.5s、2D-3T0C 每 0.1s（约 20× 实测最小 retention 安全裕量）。关键结论：2D-1T1C 在 SRAM 等面积下可达 512MB（面积效率极高）；2D-3T0C 约 2× SRAM 密度（约 64MB 上限）、32MB 时延迟最低（float node discharge 读出快）；2D-1T1C 静态功率在 512MB 时仅约 SRAM 的 87%、约同容量 Silicon-1T1C 的 30.6%；2D-1T1C 性能较 Silicon-1T1C 高约 12~18%。
  - 实验比较：材料维度（2D vs 硅）× cell 结构维度（1T1C vs 3T0C vs 6T-SRAM）× 容量维度（32MB~512MB），以及 retention-aware 机制（refresh scheduling / cyclic replacement / hot-page remapping）的消融，另与先进硅 DRAM cache 技术 BEAR [64]、TDRAM [63]、NDC [35] 对比（2D 材料收益与之正交，与 TDRAM 结合再 +5% 性能、-15.6% 能耗）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - TDM-Memory：基于修改版 CACTI（https://github.com/HewlettPackard/cacti）的 cell/array 模型，模拟 2D-material 1T1C/3T0C DRAM 的访问延迟、动态/静态能量与面积；论文未给 TDM-Memory 单独链接。
  - Hspice（Synopsys 商业工具）：大容量（32MB DRAM cache 阵列+外围电路）参考模型，验证 TDM-Memory 可扩展性（偏差 <6%）。
  - TDMSim 整体：论文声明将开源但未给链接，联网搜索未找到公开仓库，无法确认。
  - 实验验证平台：探针台 + Keysight B1500A 半导体参数分析仪 + Keysight E4990A 阻抗分析仪，测量 tape-out 的 2D 晶体管与 cache cell/array。
- 模拟器模拟什么的性能，修改了什么。
  - TDM-Memory 模拟 2D-material cell 与 array 的访问延迟、访问能量、静态功率、面积以及 retention-time 变异性（径向 retention 梯度：边缘 cell retention 短、内部长；实测 cell retention 10s-700s，大多数 >30s）。修改点：新增 gate capacitance 模型（含量子电容 C_q 与 C_og 串联，式 4）、drain capacitance 模型（SOI 式去 junction 电容，式 5）、Schottky contact resistance 模型（式 6）、variability 模型（按阵列位置取不同晶体管配置），并更新 3T0C/1T1C 的 decoder/driver（3T0C decoder 多一个输入、driver 翻倍）。
  - Hspice 参考：固定所有 device/cell 参数（由小规模 tape-out 标定、无额外调参），外围电路采用已流片验证的 Hspice 实现 [49][50]，模拟 32MB DRAM cache 阵列的访问延迟/静态功率/动态能量以对照 TDM-Memory（偏差 <6%）。
  - 大容量阵列结果：2D-1T1C 512MB 时静态功率约 SRAM baseline 的 87%、约同容量 Silicon-1T1C 的 30.6%；2D-3T0C 因 gate 寄生电容存电荷、retention 短、刷新频繁，静态功率高于 2D-1T1C；Silicon-1T1C 静态功率随容量恶化（512MB 时约 2~3× baseline）。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？
  - 开源情况：CACTI 开源（https://github.com/HewlettPackard/cacti）；TDM-Memory/TDM-Transistor（TDMSim）论文声明将开源但未发现公开仓库，联网搜索（2026-08）无法确认链接；Hspice 为商业工具。
  - 使用例子（电路配置到芯片级数据路径）：① 输入 2D 器件参数（MoS2 30nm 沟道、CGP 130nm、off-current ~10^-17 A/μm）与工艺参数到 TDM-Transistor → 得到晶体管 I-V/电容/接触电阻；② TDM-Memory 在 32MB SRAM 面积预算下做 DSE：对每种 cell 结构（1T1C/3T0C）计算 gate 电容（量子电容主导）、bitline 网络（含 Schottky 接触电阻的 RC）、decoder/driver（3T0C 双倍 driver）→ 输出访问延迟（如 2D-3T0C 32MB 延迟最低、2D-1T1C 128MB 延迟与 32MB SRAM 相当）、动态能量（大容量由互连主导、密度越高越低）、静态功率（低泄漏使刷新极少、2D-1T1C 512MB 仅约 SRAM 87%）；③ 配合 retention 地图（10s-700s 径向梯度）把刷新周期设为 0.5s/0.1s，为系统级 retention-aware 策略（中心行 1.5s、边缘 clean ways、hot-page remapping）提供参数基础。作用：在流片前量化 2D-material DRAM cache 的延迟/能量/面积与 refresh 开销，验证 2D 材料相对硅 DRAM/SRAM 的密度与能效优势，支撑芯片结构级设计决策。
