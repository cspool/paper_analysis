## 量子电容（Quantum Capacitance）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 量子电容 C_q 源于沟道有限的态密度（DOS）：向沟道注入电荷需要把费米能级相对能带移动（填充更多态），等效为与栅氧化层电容串联的一个电容 C_q = q²·DOS。对 2D 材料（原子级薄、DOS 低），C_q 不可忽略：总栅电容 C_gate = C_og·C_q/(C_og+C_q) + C_fr + C_ov（TDMSim 式 4，C_og 为单位栅宽氧化层电容、C_fr 边缘电容、C_ov 交叠电容、C_pw 多晶硅线电容），量子电容可能主导整体栅响应——即栅电容不再由氧化层单独决定。文献（Bennett & Pop, Nano Lett 2023）：0.5nm EOT 下单层 MoS2 的单栅 C_G 仅达 C_ox 的 63-78%（0.5-1V 过驱动）；EOT ≥2.5nm 时 C_q 效应被掩盖。TDMSim 在 TDM-Memory 中显式建模量子电容，使 2D cell 的位线/字线 RC 与能量估计不同于硅。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级流程：TDM-Transistor 输出晶体管参数 → TDM-Memory 计算每个 cell 的 gate capacitance（C_og 与 C_q 串联 + 边缘/交叠/多晶硅线电容按式 4）→ 影响 wordline 驱动的 RC 延迟与开关能量 → 输入到 array 级访问延迟/能量估计 → 系统级评估。例如 2D-1T1C 的访问延迟/能量在 128MB 内低于 SRAM baseline，部分归因于量子电容主导下更小的有效电容与更短互连。C_q 也是栅控存储（3T0C 电荷存于栅电容）的物理基础——其数值直接决定 3T0C 的存储电荷量与 retention 上限。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C_q = q²·DOS（2D 材料 DOS 由能带结构解析或数值给出），与界面陷阱电容 C_it 并联后再与 C_ox 串联；TDMSim 用解析公式集成进 CACTI 扩展（gate capacitance 模型）。使用要点：2D 材料存储/逻辑建模必须把 C_q 纳入栅电容（硅模型默认 C_q≫C_ox 可忽略）；对低 EOT 栅介质（<2.5nm）影响显著；TDMSim 借 2DFETs/BSIM-CMG 框架做解析化处理以支持大规模阵列仿真（对比 S2DS 等物理模拟器仅单器件、难以扩展）。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
