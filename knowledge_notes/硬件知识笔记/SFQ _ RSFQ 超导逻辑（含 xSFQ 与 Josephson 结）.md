## SFQ / RSFQ 超导逻辑（含 xSFQ 与 Josephson 结）

术语解释
以 Josephson 结（JJ）为开关元件、以单磁通量子（SFQ）脉冲编码数字信息的超导数字逻辑家族，工作于 ~4 K，切换延迟 ~1 ps 量级。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SFQ 逻辑用"有无一个磁通量子脉冲（Φ0≈2.07 fWb，对应一个约 2 mV·ps 的电压脉冲）"编码 1/0，脉冲沿超导传输线传播、由 JJ 开关产生/放大。RSFQ（Rapid SFQ）是主流时钟化家族：以电阻偏置 JJ、时钟脉冲驱动流水（web：实验验证时钟率超 100 GHz、分频器 770 GHz）；xSFQ（alternating SFQ，Tzimpragos 等，DAC'24 arXiv:2407.20942）用双轨交替编码把时钟移出门语义（每逻辑周期 = 2 个时钟周期），平均 JJ 数比 RSFQ 少 ~80%；xeSFQ 再叠加 ERSFQ 偏置实现零静态功耗（arXiv:2411.03052）。本论文用 RSFQ（时钟化、7 级流水 PU）实现高吞吐处理，用 xSFQ（无时钟、sub-GHz）实现 ENC 编码单元以省 JJ 与功耗。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文的流水结构：PU 用 RSFQ 门组成 7 级流水线，每级处理 syndrome 位流中不同位置（移位寄存器/PTL 行缓冲的抽头 + 组合真值表），10 GHz 假设（保守，文献 16 GHz、本文 DLM 实测 33 GHz）；ENC 用 xSFQ 门实现减法/移位/unary 计数，速率匹配 1 Gb/s 电缆（sub-GHz）。选型逻辑链：SFQ 处理（10 GHz）≫ 电缆串行化（1 Gb/s），因此用流式共享硬件替代全并行（Clique 每 ancilla 专用硬件 ≥96 JJ），处理延迟（≤17 ns）被串行化延迟（500 ns）隐藏；时钟化与无时钟混合取各自优势——时钟化适合长流水吞吐、无时钟适合低频省功耗（JJ 0.2 aJ/次开关 + 50% 偏置开销，IcePack 功耗 10–42 nW/ancilla）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Nb 工艺（Nb/Al/AlOx/Nb 三层 JJ），MITLL SFQ5ee 支持 >100 万 JJ/cm²；典型单元 DRO/NDRO（破坏性/非破坏性读出）、移位寄存器（16 GHz 演示、误码 <10^-10）、merger tree（异步 OR 树）。本论文以 JJ 数作面积一阶代理：平均 4 JJ/ancilla（PU 1502 恒定 + PPU/ENC 随块数变化，3.2–13.6 范围），Clique 需 ≥96；移位寄存器 vs PTL 存储每 bit JJ 差 40×。使用场景：4 K 层近量子处理器件（控制、读出、压缩、近似解码），热预算 1 mW/qubit 下以 aJ 级能耗换取电缆带宽。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
