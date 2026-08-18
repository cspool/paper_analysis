## 保持时间（Retention Time）与径向保持时间梯度（Radial Retention Gradient）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Retention time 是 DRAM 单元在无刷新情况下维持所存电荷/数据有效的最长时间，由存储电容泄漏（通过访问晶体管 off-current）决定；刷新周期必须 ≤ 最小 retention 以保证数据完整性。硅 DRAM 标准 retention 约 64ms（JEDEC DDR4），2D 材料因 off-current 低至 10^-17 A/μm 而可达 10s-700s（TDMSim 实测）。关键实测发现：2D 阵列存在径向保持时间梯度（radial retention gradient）——阵列边缘的 cell retention 明显短于内部（边缘 10s 量级、内部至 700s，绝大多数 >30s），推测源于光刻邻近效应、接触电阻变化与局部互连寄生加重边界 off-state 泄漏。径向梯度与硅 DRAM 的"弱 cell 随机分布"（VRT、RAIDR 假设）本质不同，使传统按行处理弱 cell 的 retention-aware 刷新不适用。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级流程：实测 retention 分布（Fig.12a 空间 retention 地图、Fig.12b 分布直方图）→ 若按最弱 cell 保守统一刷新（0.5s、约 20× 安全裕量、覆盖 360K 温度与 process tail cells）则浪费长 retention 空间 → retention-aware 策略按径向梯度分级：边缘行 0.5s 刷新、中心行延至 1.5s；row 内物理 retention tier 为最小粒度，边缘弱 tier 组成的 cache ways 限定为 clean ways（只放指令/预取等与主存一致的 clean block，配合 per-row validity counter 检测违例并重填），tag 严格映射非循环 way；3T0C 因 gate 电容存储 retention 更短（0.1s 刷新）。该机制把访问干扰率降 75.6%、refresh energy 降 65.4%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：测试平台（探针台 + Keysight B1500A/E4990A）测 off-current 与 retention 分布；建模上 TDM-Memory 的 variability 模型按 cell 在阵列中的位置取不同晶体管配置（边缘访问更长延迟、静态功率按配置分布加权）。使用要点：2D 材料评估必须实测 retention 空间分布而非假设均匀；刷新周期 = 最弱 cell retention × 安全裕量，径向梯度决定刷新分级与数据放置（hot 数据避开高刷新区）；论文用 Murphi 模型检查器验证策略正确性。相关已有条目：本库"DRAM Retention Failure 与 True-/Anti-Cell""Variable Retention Time（VRT）"覆盖硅 DRAM 侧，本条覆盖 2D 材料特有的径向梯度场景。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
