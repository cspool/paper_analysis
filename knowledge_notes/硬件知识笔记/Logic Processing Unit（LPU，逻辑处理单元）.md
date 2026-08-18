## Logic Processing Unit（LPU，逻辑处理单元）

术语解释
自行车架构中附着于 BB 码块的 surgery ancilla 系统，提供原生逻辑 Pauli 测量能力；由 Cross、He、Rall、Yoder 2024 年（arXiv:2407.18393）引入。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LPU 是广义 code surgery 的 BB 码实例：利用 BB 码对称性构造的、以最小额外物理 qubit 数最大化测量能力的 ancilla 系统。gross 码的 LPU 用 90 物理 qubit、two-gross 的用 158。结构：LPU 独立连接 pivot $L_0$ 与其 ZX-对偶 $L_6$ 两个逻辑 qubit，内含分离的 X 模块与 Z 模块，可测 $X_{L_0},Z_{L_0},X_{L_6},Z_{L_6}$ 的任意乘积；共提供 15 种双逻辑 qubit 原生 Pauli 测量，与 36 个 shift automorphism 组合得 540=15×36 个 native 多体测量（最多 12 逻辑 qubit）。X 模块与 Z 模块经瞬态 bridge 耦合可实现跨 pivot 与对偶的 $X\otimes X$ 测量，移除 bridge 则同一逻辑周期内完成两个单 qubit 测量 $X\otimes I$ 与 $I\otimes X$——同时测量可减少深度、摊销 LPU 成本。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（本论文 pivot 注入方案）：①adapter 的 inter-module 测量把 |T⟩ teleport 到 pivot $L_0$；②LPU 在 $L_0$ 上做 in-module 测量把魔法态传到目标逻辑 qubit；③条件 Clifford 校正吸收进 LPU 对 pivot 的 X/Y 测量；④协议各旋转经 automorphism 共轭 + LPU 单次测量完成；⑤最终 parity check 与 postselect 也走 LPU 测量。成本模型：in-module 测量 τ=120（gross）/216（two-gross）时间步，总误差含测量结果翻转 $p_{\rm meas}$ 与逻辑 depolarizing 两部分，λ=p_meas/p_intra。关键设计权衡：LPU 测量比 automorphism 慢且错误率高（10⁻⁵ vs 10⁻⁶·⁴ @ gross、10⁻³），故本论文的编译目标 = 最小化 LPU 测量次数。Y 基 pivot 测量同时占用 X/Z 双模块，双轨并行时须串行（出现概率 3/4）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 泛化形态是 extractor（arXiv:2503.10390）：QLDPC 架构中实现任意逻辑 Pauli 测量的 surgery 系统（EAC 块经 bridge 连接）；"full extractors" 支持任意逻辑 Pauli 算子测量且免去此前编译开销（HGP 码）。工程实现：bicycle_cliffords Rust crate 提供自行车架构 Clifford 合成；Relay-BP 解码器面向 FPGA/ASIC 实时实现；本论文假设 LPU 仿真错误率（[27] Table I）未来随解码器改进而下降，工厂性能随之提升。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
