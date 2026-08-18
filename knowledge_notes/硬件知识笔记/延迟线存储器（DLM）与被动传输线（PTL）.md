## 延迟线存储器（DLM）与被动传输线（PTL）

术语解释
以超导被动传输线为延迟介质、让 SFQ 脉冲循环驻留实现的顺序访问存储；"可寻址的环形移位寄存器"，数据位不需 JJ 存储单元。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DLM 用 PTL（无源超导微带/带状线）的低衰减、低色散特性让脉冲以一定周期在环路中循环，控制器在读写窗口同步提取/重写，实现无 JJ 存储单元的顺序存储。高 kinetic inductance 材料（NbN 纳米线等）的单位长度延迟极高，从而把位间距压到很小（web：Tzimpragos 等的 Pulsar 工作 arXiv:2205.08016；Scientific Reports 2023 "Addressable Superconductor Integrated Circuit Memory from Delay Lines"：20–100 GHz 工作、SC2 工艺 10s Mbit/cm²、支持寻址/重写/非破坏读出）。本论文把 DLM 用于 PU 行缓冲与 TCU 预测存储：10 GHz、20% 单元/1% PTL 时序变异裕量下每 PTL 可存 41 bit 才需插入 DRO 分段防抖动积累，JJ/bit 比移位寄存器少 40×。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在本论文 PPU/PU 中的角色：SCU 行缓冲用短延迟 PTL（延迟 = 相邻 index 间隔，数拍）；TCU 预测存储用环形 PTL（延迟 = 整测量轮 ~1 μs 的等价位距，避免超长移位寄存器或前馈 PTL）。运转流程：脉冲写入 → 环路循环 → 同步控制器在正确相位读/写（重写=保持，丢弃=擦除）→ 读出转发到组合逻辑。同步控制器保证时钟相位对齐，DRO 单元按需分段插入修正累积 skew。面积由控制器速度、线速度因子、布线层数、线宽/间距决定：Nb stripline（SFQ5ee）≤3000 μm²/ancilla，MoN stripline（SC2）187 μm²/ancilla（50 万 qubit/cm²）；JJ 与 PTL 分层放置后 PTL 足迹即有效面积。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Nb 或 MoN 带状线布线层 + 同步控制器 + 分段 DRO；本文原型 2 mm Nb 延迟线环（MITLL SFQ5ee 流片）实测 33 GHz 循环存储 2 个 SFQ 脉冲，模拟电压读出幅值与脉冲数成正比（非破坏性读出）。使用方式：顺序访问存储（环形缓冲、FIFO、行缓冲、预测存储），规避超导随机访问存储（JJ 阵列）的面积/功耗劣势；与移位寄存器等价但省 JJ 与功耗，代价是控制器时序收敛难度（需 DRO 分段）。扩展：更高速度、高 kinetic inductance 先进工艺（imec/MITLL）、多 SFQ 芯片 bonding。

涉及论文标题：
- A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin
