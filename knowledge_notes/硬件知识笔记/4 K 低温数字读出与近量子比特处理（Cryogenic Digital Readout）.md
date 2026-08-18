## 4 K 低温数字读出与近量子比特处理（Cryogenic Digital Readout）

术语解释
把量子比特读出数字化并尽可能靠近量子处理器（mK–4 K）完成，用数字化数据流替代逐 qubit 模拟电缆，减少 mK↔300 K 线缆数量；由此引入 4 K 层处理机会与热/带宽约束。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
可扩展量子系统趋势：下游（300 K→mK）控制流量已被低温微波脉冲整形或 SFQ 脉冲串大幅削减（可降几个数量级），剩下上游（mK→300 K）周期性测量数据成为带宽与热负载瓶颈。数字读出器件（如 Josephson photomultiplier JPM：JJ 偏置在临界电流附近把 qubit 态映射为腔光子占有、输出二值 click/no-click，web 佐证：Howington/Opremcak/McDermott 等 IEEE TAS 2019，JPM 输出可直接接 SFQ 逻辑）使每 ancilla 每轮仅 1 bit。数字位经串行化后可共享电缆，但受 1 μs 测量轮限制：加电缆/提速增加热负载（电缆 1 mW/Gb/s + 10.5 mW 外设，约 0.1 mW/ancilla，占 1 mW 预算大头），延长串行化则超轮限——因此 4 K 层压缩（IcePack）是唯一同时降热负载与延迟的路径。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文的读出链：mK 层数字读出（每 ancilla 1 bit/轮）→ 4 K 层 IcePack tile（PPU 全零块过滤 → PU 空间/时间聚类 → ENC 变长编码）→ 单根 1 Gb/s 不锈钢同轴电缆上行 300 K → 主机流水解压（2.5 ns）→ 全精度解码器。4 K 层约束塑造架构：热预算迫使少 JJ（平均 4 JJ/ancilla）、低功耗逻辑族（xSFQ 用于低频 ENC）；速率失配（10 GHz 处理 vs 1 Gb/s 电缆）使流式优于全并行。对比三个读出方案（论文图 1b）：无处理基线（数据全发）、全 SFQ 解码（4 K 解码精度损失）、层次化解码（Clique 逻辑错误率高 1000×）；IcePack 压缩解码全部留在 300 K，0 精度损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现路线：JPM/SFQ 读出接口、cryo-CMOS/cryo-FPGA 读出（web：4 K SiGe 接收机 6 mW 实现 1 μs 内 >98% 保真度读 transmon；4 K–300 K 数据链有 40 Gb/s 同轴、56 Gb/s 光学、260 GHz 背散射 176 fJ/b 等方案）。本论文假设 1 Gb/s 同轴为保守基线（10× 低于 SFQ 处理速度）。使用方式：按 tile 参数化（每 tile 数百–数万 ancilla，块数与错误率相关），多 tile 共享单电缆，适配 lattice surgery / 动态码距（CaliQEC、Q3DE）等运行时变化。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
