## Bicycle Architecture（自行车架构 / Tour de gross 模块化 FTQC 架构）

术语解释
IBM 基于 BB 码 + LPU 提出的模块化容错量子计算架构（Yoder et al., "Tour de gross", arXiv:2506.03094）；本论文的蒸馏工厂是其 T-factory 核心组件。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 逻辑信息编码在多个 QLDPC 码块中，每块附一个 extractor/LPU 型 surgery ancilla；各 ancilla 经 bridge 互联以支持灵活的多逻辑算子测量，为通用 FTQC 再连接 magic state 工厂。指令集（有已知 timestep 与逻辑错误率）：idle、shift automorphism、in-module/inter-module 测量、T 态注入。相对 surface code 的卖点：每个物理 qubit 可执行 ~10× 更大的逻辑电路；IBM 路线图 Starling（2029，200 逻辑 qubit、1 亿门）与 Blue Jay（2033，2000 逻辑 qubit、10 亿门）。架构满足六准则：逻辑保真度、可寻址性、通用性、自适应性、模块化、效率。adapter 是连接魔法态培养与 BB 内存的 surgery ancilla 系统，使任何培养方案都能接入本架构。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 魔法态工厂在架构中的运转（本论文两级工厂）：surface-code MSC 培养（454 qubit，输出 p_in≈10⁻⁶）→ adapter inter-module 测量注入 two-gross 块 → 块内 15-to-1 蒸馏（734 qubit，15 个旋转 + 4 个 X 基检查）→ 输出 |T⟩ 经 Z⊗Z 测量投递到消费逻辑 qubit。资源：二级 τ=11080 时间步、体积 8.1×10⁶，p_out≈4.1×10⁻¹²（p_phys=10⁻³）或 ≤10⁻¹⁷（10⁻⁴）。架构瓶颈分析：inter-module 通信是主要瓶颈（外部工作 syn@fac 把旋转合成搬到工厂侧使电路失败概率 ~9× 下降）；工厂设计分解为 operation-limited（in-module 旋转主导）与 source-limited（输入+注入主导）两区，理想工作点让两类误差相当。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 编译侧：图形化语言编译 unitary+measurement 电路到自行车指令集；本论文的 mapper/masking/TSP/双轨优化与协议压缩即工厂专用编译流水线。评估侧（arXiv:2604.20013）：早期容错自行车架构的能力与瓶颈评估提出 transvection 式 Clifford 延迟与 Clifford 插入。硬件侧：长程耦合器（L-coupler）提供跨块纠缠。web 补充：allaboutcircuits 对 IBM 路线图的报道确认 gross/two-gross 与 Starling 的对应关系。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture
