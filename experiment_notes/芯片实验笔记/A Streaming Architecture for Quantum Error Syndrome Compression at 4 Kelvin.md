## A Streaming Architecture for Quantum Error Syndrome Compression at 4 Kelvin

- 属于芯片设计的实现是什么？实验比较什么？
  - 近似匹配（组件级原型 + 工艺级存储结构分析，非完整芯片流片）：实现为基于被动传输线（PTL）的延迟线存储 DLM——将多层互连的布线层用作无 JJ 数据存储介质（"可寻址环形移位寄存器"，数据靠传输线往返循环驻留，同步控制器 + DRO 分段防抖动），存储密度来自高 kinetic inductance 线的单位长度延迟；流片验证 2 mm Nb 延迟线存储环（MITLL SFQ5ee 工艺），实测以 33 GHz 循环存储 2 个 SFQ 脉冲。实验比较：PTL/DLM vs 每 bit 用同步 DRO 单元的移位寄存器存储（JJ/bit 少 40×）；Nb stripline（SFQ5ee，≤3000 μm²/ancilla）vs MoN stripline（SC2，187 μm²/ancilla、支持 50 万 qubit/cm²）两种工艺参数；IcePack 平均 4 JJ/ancilla vs Clique ≥96 JJ/ancilla。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 无专用工艺/版图模拟器：PTL 面积由 MITLL SFQ5ee [67] / SC2 [66] 工艺 stripline 规格（线宽、间距、速度因子、布线层数）结合 DLM 研究 [75] 外推；功能正确性由 PyLSE（https://github.com/UCSBarchlab/PyLSE）门级仿真验证。论文未说明修改任何模拟器。
- 模拟器模拟什么的性能，修改了什么。
  - 工艺级分析计算 PTL 存储面积足迹（由控制器速度 10 GHz、线速度因子、布线层、线宽/间距决定）与每 ancilla 的 JJ/PTL 资源分配；未报告模拟器修改。
- 开源情况。
  - 芯片网表/版图未说明开源；算法 artifact 开源（Zenodo https://doi.org/10.5281/zenodo.19446086，CC BY 4.0）。基于论文描述的使用例子：MITLL SFQ5ee 工艺支持 >100 万 JJ/cm²，IcePack 最大配置（p=10^-4、T=500 ns、97,250 ancilla/tile、约 31 万 JJ/tile）远低于单 cm² 集成能力；PPU/PU 存储器由 PTL 替代 DRO 移位寄存器实现，10 GHz、20% 单元/1% PTL 时序裕量下每 PTL 存 41 bit，JJ 与 PTL 分层放置使 PTL 足迹成为有效面积估计；扩展路径为更高速度、高 kinetic inductance 先进工艺（imec/MITLL 多层堆叠）与多 SFQ 芯片 bonding。
