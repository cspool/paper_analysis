## Monte Carlo 错误注入评估（SE/16E/32E/DQE/DQSE 场景）

术语解释
用大规模随机错误注入统计 ECC 方案的纠正/检测覆盖率：按故障位置与场景翻转指定位、逐方案译码分类 CE/DUE/SDC、百万至千万次迭代聚合概率，是 ECC 论文的标准可靠性评估方法。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Monte Carlo 错误注入：对每个 (方案, 场景) 组合执行 N 次迭代（本文 1000 万次），每次在指定位置独立以 50% 概率翻转指定数量的比特，随后运行该方案的（分层）译码流程，把结果分类为 CE（correctable）、DUE（detectable but uncorrectable）或 SDC（silent data corruption，含 DME/UUE），最后按迭代次数聚合出 CE/DUE/SDC 百分比。本文的场景空间基于 DRAM 现场故障研究（DDR4/DDR5/HBM 现场数据分析）抽象为三个错误位置：In bank（bank 内部：SE 单元/BLSA 错、16E CSL/SWL 错、32E SWD 错、SE+SE 双单错）、Write link（写传输路径：SE、DQE 数据 pin 错、DQSE 数据选通错）、Out bank（读路径 bank 之外：SE、DE 双位错（TSV 等外围）、DQE、DQSE 读链路错）。对链路错误，DUE 按可重传/重试折算为 CE（这是 L-ECC 语义）。
从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
评估器模拟的是"故障 → 各层译码器行为"的闭环，而不是周期级硬件：例如 Cerberus 的 in-bank 16E 场景，注入 16 个连续位错 → Decoder2（H2 SEC）miscorrect 可能发生 → Decoder3（H^S-ECC SSC）把该符号整体纠正 → 输出 CE=100%；HBM4 同一场景下 O-ECC 16-bit SSC 可纠，但 SE+SE（两符号各一位错）时 O-ECC 只有两错同符号才纠、否则残留模式超出 S-ECC 能力。多位置场景（如 In bank+Write link 的 32E+DQSE）检验各层同时接敌时的协作——本文 Table II/III 的结论：Cerberus 32b 在全部单/多位置场景下 CE 约 100%（仅 32E 类超大簇错为高检测 + 少量 SDC），LPDDR6 对 16E 只有 0.048% CE、HBM4 对 out-bank DQE 约 50%——这正是分层共享冗余 vs 分层独立冗余的量化差距。硬件架构评估上的配套：错误注入定可靠性，Accel-Sim（tCL/tWL 时序调整）定性能，DC 综合定面积/能耗，三者共同构成 ECC 硬件论文的评估三角。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：轻量 Python/C 模拟器即可（逐场景穷举或随机注入 + 矩阵译码，无需 RTL）；本文的实现未开源（论文未给链接，无法确认）。使用要点：① 迭代次数要覆盖尾部事件（本文 1000 万次使 10^-5 级 SDC 率可统计；HBM4-CRC 的 SDC 低至 10^-6 量级）；② 错误模型必须锚定现场故障数据（SE/DE/16E/32E/DQE/DQSE 分别对应 BLSA/TSV/SWL/SWD/DQ/DQS 的物理失效域，宽度 8–32 位）；③ 对链路类错误把重传语义折算进 CE，才能公平对比单层与多层方案；④ 与 bounded-fault、检测概率等码性质评估互补——错误注入回答"整体覆盖率"，码性质分析回答"最坏情形保证"。
HBM-CASO 补充视角（ISCA'26，HBM 可靠性评估方法论）：沿用同一套"大规模随机注入 + 逐方案译码分类"范式但规模与场景不同：(1) data-block 级覆盖率——用 SBF/BBF/WBF/SAF 四类故障组合出 15 种代表模式，每模式 10^9 次 Monte Carlo 注入（burst 内每位 50% 概率翻转、至少 1 位翻转，模拟最坏随机损坏），译码分类 DCE/DUE/SDC（Table III）；传输检测单独报 UE%（Table IV）——R/G-mode 读传输用重建 regional/global 码字验证、写传输用 HBM 与控制器两侧 XOR 累加结果比较（不匹配整批重传），G-mode 全部场景 UE%=0。(2) 六年前景（lifetime）评估——用 fault-mode-aware 注入框架（按 DUO [18] 与 RATT [11] 的建模法，FIT 率与故障类型分布取自 HPC 现场 DRAM 研究 [17][72]），遵循 FaultSim [50] 操作模型：3 小时注入/检查间隔、12 小时 scrubbing、transient 被清扫清除、permanent 持续累积，每配置模拟 10^12 cacheline（事件驱动引擎跳过无故障时段）；两种配置：permanent-only（p=10^-4, t=0）与 mixed（p=10^-5, t=10^-5），p/t 按 FIT 率保守放大 10× 反映 HBM 脆弱性。结果：G-mode 长期 DUE 1.6×10^-3、SDC 1×10^-10（mixed 下 5×10^-12），优于 baselines/Config-ECC/COMET 并与最强 prior（Domain-ECC DUE 最低、DUO SDC 最低）有竞争力。评估三角与 Cerberus 相同：错误注入定可靠性、Ramulator2（tCL 调整）定性能、45nm DC 综合定面积、DRAMSim3 定功耗。
RangeGuard 补充视角（ISCA'26，RID 语义保护下的故障分类）：RangeGuard 用同一套"大规模随机注入 + 逐方案译码分类"范式，故障模式取 SE（单比特）、DAE（双相邻）、16E（16-bit 边界内）、32E（32-bit 边界内，SWL/SWD 类）、FC（全块）五种（来自 DDR5/HBM 现场研究 [5][42]），单故障与双故障组合（SE+SE/SE+DAE/SE+16E/SE+32E）每场景随机选位置、区域内每 bit 以 50% 概率翻转，每实验 10^9 次；对 RangeGuard 把"纠正"分类为 Bounded Error（BE，有界近似恢复）而非 CE。DNN 准确率实验按 DDR5 现场分类比例加权注入五种模式（Table IV：SE 0.009×BER、DAE 0.023×BER、16E 0.175×BER、32E 0.793×BER），PyTorch 推理期间注入权重与激活、逐方案 ECC 译码、仅在不可纠或有界纠正时更新 tensor，每 BER×方案 100 次。对比 Cerberus 的三位置（in-bank/write link/out-bank）场景空间，RangeGuard 简化为块内故障模式空间，但新增"语义层"分类——intra-range 错误（不改 RID）不计入纠错预算、可无限量放行，inter-range 错误才消耗 RS 纠错符号。
涉及论文标题：
- Cerberus: Cross-Layer ECC Co-Design for Robust and Efficient Memory Protection
- HBM-CASO: A Coordinated Approach to HBM System-Level and On-Die ECC
- RangeGuard: Efficient, Bounded Approximate Error Correction for Reliable DNNs
