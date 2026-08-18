## CACTI 与 TDM-Memory（2D 材料 cell/array 建模扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CACTI 是 HP 实验室（Hewlett Packard Labs，Muralimanohar/Balasubramonian/Jouppi 等）开发的架构级缓存/内存建模工具：给定容量、block size、associativity、工艺节点与 cell 类型，基于解析 RC 与电路模型估计访问延迟、动态/静态能量与面积（CACTI 6.0 论文、开源 https://github.com/HewlettPackard/cacti）。TDMSim 的 TDM-Memory 以 CACTI 为 baseline 框架，针对 2D 材料做扩展：按指定 memory type 与 TDM-Transistor 参数做电路级设计空间探索，选 cell 技术（1T1C/3T0C DRAM）并输出最优组织的能耗/访问延迟/面积。CACTI 变体谱系：NVSim（非易失内存）、DESTINY（3D NVM/eDRAM）、CACTI-3DD（3D die-stacked DRAM，本库已有条目）、FN-CACTI（FinFET）——但均不含 2D 材料特性（Table I 对比）。
- 从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- TDM-Memory 的四项核心扩展：(1) Wordline Gate Capacitance Model——gate 电容 = (C_og·C_q/(C_og+C_q) + C_fr + C_ov)·W + C_pw·L_phy（式 4，量子电容主导，见量子电容条目）；(2) Bitline Drain Capacitance Model——2D 材料难掺杂致源漏界面耗尽区扩大、结电容被抑制（SOI 式），排除 junction 电容仅保留 C_fr、C_ov 与金属互连电容 C_metal（式 5）；(3) Bitline Schottky Contact Resistance Model——式 6 电压相关接触电阻并入 bitline RC（见 Schottky 条目）；(4) Variability Model——TDMSim 从 TDM-Transistor 取多种晶体管配置（传统 CACTI 只用单一配置），按 cell 在阵列中的位置分配（边缘访问更长延迟），静态功率按配置分布加权。此外 3T0C/1T1C 的 decoder/driver 差异化更新（3T0C decoder 多一输入、driver 翻倍）。验证：cell 级对 100 个流片样本测访问延迟/能耗（Fig.7）；array 级对小规模流片阵列测时序/变异性（Fig.8a），32MB 大阵列用 Hspice 参考对照（固定小规模标定参数、无额外调参，偏差 <6%，Fig.8b）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CACTI 为 C++ 解析模型（输入容量/组织/工艺参数，输出延迟/能量/面积）；TDM-Memory 在其基础上按上述公式与 variability 逻辑扩展，输入接 TDM-Transistor 输出、输出接 gem5 系统模拟器（标准化接口使系统模拟器自动获取 2D DRAM cache 参数、免人工处理）。使用要点：CACTI 系工具适用于"任意容量/组织"的快速设计空间探索（TDMSim 用 32MB SRAM 面积预算扫描 32-512MB 的 1T1C/3T0C 配置）；TDM-Memory 相对 Hspice 在 32MB 规模偏差 <6% 证明可扩展到任意阵列大小；TDMSim 本身论文声明将开源但联网（2026-08）未找到公开仓库，无法确认链接。
涉及论文标题：
- TDMSim: Enabling High-Density and Energy-Efficient GPU DRAM Caches with 2D-Materials for Data-Intensive Applications
