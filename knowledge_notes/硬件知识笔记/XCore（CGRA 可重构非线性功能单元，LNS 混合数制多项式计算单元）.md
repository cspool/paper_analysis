## XCore（CGRA 可重构非线性功能单元，LNS 混合数制多项式计算单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- XCore 是 LoRA（ISCA'26，复旦大学）提出的轻量可重构功能单元，集成到 CGRA 的 XCore-PE 中，用于高效计算非线性函数（激活函数、三角函数、幂/对数）。它采用普通算术+LNS 混合数制：加/减用普通算术（简单），乘/除/幂/对数/多项式在 LNS 中完成（把乘法变成加法、幂变成乘法）。32-bit 数据，支持可编程定点（Qm.n，fraction 宽度可调）、FP32、LNS 三种格式（LNS 对 0/负值未定义，用绝对值做对数变换、零与符号编码在额外 bit 中）。四种可配置模式：①乘/除法、②幂 x^y、③对数 log_b(x)、④多项式（最多 6 项=5 个含变量项+1 个常数 bias 项）；前四种模式 4 cycle，多项式模式 7 cycle。
- 架构（Fig.2，5 级流水，级间寄存器分隔）：Pre-Process（把输入 x 与 breakpoints 比较、查 LUT 取每段多项式参数 log2(c_i)/k_i/bias；bias 可为常数或来自其他功能单元的输入 y）→ LOG（对数转换器把 x 转 log2(x)；定点输入经 LOD 检测 MSB、浮点输入直接取尾数，APP 单元 PWL 近似 log2(1+f)，APP 由 SOTA 探索框架 [82] 生成）→ LNS（可编程 30b×30b 乘法器：5 个 Booth 编码器生成 30b×6b 部分积求和；6 项多项式时把 30b 操作数分解为 5 个 6b 子操作数算 log2(c_i)+k_i·log2(x)，因 k_i 指数只需 6 bit；乘/除模式中 y 也在此级做对数变换以复用乘法器中空闲加法器）→ ALOG（CPA 算 log2(x)±log2(y)，SAT 检测饱和/溢出并对齐格式，反log 转换器 PWL 近似 2^f 转回普通算术）→ Output（乘法/除法/幂/对数模式直接输出；多项式模式用加法树把各普通算术项求和，CPA 可配定点/浮点加，输出加寄存器改善时序）。
- 硬件参数：Verilog 建模（其余 SoC 用 Chisel），TSMC 40nm 综合；XCore-A/B/C 三个变体（7 段，510/485/510MHz，面积 71.7–78.4k µm²、功耗 16–17.4mW）与 XCore-C（6 段）；对比 huicore（CORDIC 通用复杂函数加速器，28nm，153k µm²、≥20 cycle）与 Flex-SFU（分段二次逼近，28nm，22.9k µm² + MAD 单元）、PACE（3-term Chebyshev，12nm，~32k gate）：XCore 以更低硬件开销支持更多格式、x^y 与复合函数单步逼近。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（一次 6 项多项式逼近 tanh(x)+1 的执行）：输入 x 进入 XCore → Pre-Process 与分段 breakpoints 比较选中子区间、从 LUT 读出该段参数（log2(c_i)、k_i、bias）→ LOG 级 LOD 检测 MSB 得 m、APP 近似 log2(1+f) 拼出 log2(x) → LNS 级乘法器配置成 5×(30b×6b)：并行算 5 个 log2(c_i)+k_i·log2(x) → ALOG 级 5 路反log 转回普通算术（2^(m+f)=2^m·2^f）→ Output 级加法树（CPA 可配定点/浮点加）把 5 项与 bias 求和，7 cycle 输出。乘/除/幂/对数模式 4 cycle：LOG → LNS（乘/除中 y 也在 LOG 转 log2(y)）→ ALOG（CPA 加减 + SAT）→ Output 直接输出。复合函数（tanh(x)+1、sin(x)+cos(x)、ln(sin(x))）由一个 XCore 直接逼近，无需分解成多个运算单元。
- 在 CGRA 中的作用：一个 XCore 节点顶替 PICACHU 的多个 Taylor MAD 节点——Swiglu DFG 从 37 节点（PICACHU）降到 15 节点（LoRA，含 1 个 XCore），释放 PE 供 loop unrolling；sqrt 等 PICACHU 不支持的函数由 XCore 在片上执行，不再卸载 CPU（KNN 中 CPU EXE 周期占比极大是 PICACHU 的痛点）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL（可综合，论文声明生成 Verilog 可用于 ASIC 实现或 FPGA 原型）；APP/反log 的 PWL 近似单元由约束驱动参数探索框架 [82] 生成，在目标精度与硬件开销间折中；对数/反log 转换器的精度决定 implementation error（误差分析：c_ix^(k_i)=2^(log2 c_i + k_i log2 x)，转换误差 δ/β 被放大 k_i·ln2·δ 倍）。XCore 配置由软件（PiecewiseChebFitter，Python）生成的 breakpoints/系数/次数写入 LUT。
- 使用：在 LoRA 工具链中，用户用 #pragma 标注 loop kernel 与自定义非线性函数（__CGRA__HARDWARE_OP），编译器自动生成 XCore 配置并映射进 CGRA；XCore 变体选择（XCore-A/B/C，不同转换器精度/段数/项数）按归一化 ADPP-精度权衡（Fig.8 设计空间探索）；评估精度指标 AAE/sq-AAE/MSE 与软件逼近（LoRA-SW）对比，误差同数量级（sin 受对数转换器精度 ~1e-5 上限限制）。开源：https://github.com/Dai-dirk/COFFA/tree/LoRA-ISCA-AE（Verilogs 目录含 SoC 与 XCore-A/B/C 的 Verilog）。

涉及论文标题：
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
