## Computation simplification optimizations（计算简化优化：zero-skip / digit-serial / bit-serial）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 计算简化（computation simplification）优化是微架构为省能耗/提性能对"平凡"计算做旁路或加速的一类数据相关优化：检测到操作数为特定值（0、1、全 1 等）时跳过或缩短算术执行。Helium 论文（§I 引 [10][43][46][52][53][97]）把零跳过（zero-skip）、早期终止乘法等列为威胁 CT 编程的新数据相关优化来源；µobs functions 正是为这类优化建模。三类具体实例：① zero-skip（及其变体 zero/all-ones-skip、zero/one-skip）——操作数含 0（或 0xFF..F、1）时走快速 µobs；② digit-serial multiplication——乘法器按乘数最高字节是否为 0 分多档执行周期（Hartley & Corbett 的 digit-serial 处理，论文引 [43][46]）；③ bit-serial division——位串行除法单元（如 CVA6 RISC-V CPU 的除法器，论文引 [99]）按两操作数前导零差分 65 个 µobs。
- 逻辑链：这些优化使算术指令（传统 CT 认为"安全"、可放心处理秘密）成为 intrinsic transmitter：操作数值 → 执行周期数不同 → 时序可观测 → 秘密泄漏。Helium 用 µobs functions 把它们抽象建模，量化泄漏；cio（[37]）则针对表 V 的计算简化类别（ADD/SUB/MUL/OR/AND/XOR/SHL/SHR/SAL/SAR 的 mul64/cs64/cs32）做零泄漏二进制码变换。
- Web 证据：CVA6（https://github.com/openhwgroup/cva6，论文引 [99]）是 bit-serial 除法 µobs function 的硬件来源；digit-serial 处理技术（Hartley & Corbett, IEEE Trans. Circuits and Systems 1990，引 [46]）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在硬件架构中，计算简化优化的运转：MUL 进入功能单元前检测操作数——任一操作数为 0/1 时，乘法器旁路/缩短路径，指令在 µobs1（快速）路径完成；否则完整执行 µobs2（慢速）。digit-serial 乘法器把乘数按字节分档（op2[8:31]==0 → µobs1；op2[16:31]==0 → µobs2；op2[24:31]==0 → µobs3；否则 µobs4），高字节全 0 的窄操作数省周期；bit-serial 除法器逐位串行，所需周期取决于 (op1>>d) 与 op2 的关系（前导零差），最多 65 个 µobs。
- 具体例子（Helium 的评估）：Case Study II——Firefox feConvolveMatrix SVG 卷积滤波处理 3 个单字节像素（黑/白，3-bit 秘密）：zero-skip 下 8 条 µtrace 每条 PML=3（全泄漏，因为每个像素值触发不同 µobs）；digit-serial 下恒单 µtrace、PML=0（单字节操作数恒走同一 µobs 档）——同一优化在不同程序下安全性反转（Poly1305 下 digit-serial 泄漏高）。Case Study IV——cio 的表 V 类别（ADD 两操作数 {0}、MUL 两操作数 {0,1}、OR/AND/XOR {0,0xFFF}、SHL/SHR/SAL {0} 等），unsafe 值触发"快速路径"。
- Annotations：µobs 是抽象路径而非具体周期数（可标注为 µobs0/µobs1 等不透明标签，保护厂商细节）；同一优化可用不同粒度 µobs function 建模（Case Study III 把 digit-serial 改成 8/4/2/1-bit 宽度以变 1/2/4/8 个 µobs）；优化对安全的影响必须程序×微架构对评估。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件前端在指令进入执行单元前检测操作数是否为"平凡"值（比较器），命中则旁路/提前终止。µobs functions 以伪代码/约束公式形式由 Helium 用户提供（artifact 内置 zero-skip、digit-serial、bit-serial 及 mul64/cs64/cs32 类别），经 Tracer 分析程序泄漏；厂商可用 RTL 工具（SynthLC 类）自动合成。
- 使用：设计者用 Helium 判断"启用某计算简化优化对某程序是否引入可接受泄漏"——如 Poly1305+zero-skip 泄漏极小（P[ℓ≤1.35×10⁻⁹]≥1−9.39×10⁻¹⁰），可放弃昂贵缓解；Poly1305+digit-serial 泄漏高（P[ℓ≤0.97]≥0.51），需缓解。限制：µobs functions 只建模 intrinsic transmitter；动态/静态 transmitter（store buffer forwarding、prefetch 等）需要维护抽象微架构状态，列为未来工作（§VIII）。

涉及论文标题：
- Helium: Quantifying Microarchitectural Side-Channel Leakage with Probabilistic Guarantees
