## LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

- 属于硬件架构的实现是什么？实验比较什么？
  - **LogART AE（Arithmetic Element）硬件设计**：针对 DLog 量化权重 Ŵ 与激活 X 的乘法计算（Ŵ·X），设计了无乘法器的 AE 架构。核心原理：
    1. 基-2 对数域的 Ŵ 与 X 相乘等价于移位操作（X << Q_W），完全消除乘法器。
    2. 基-√2 对数域通过 HAF（Hardware Approximation Function）用 K-term Signed Dyadic Expansion (SDE) 近似 √2，将乘 √2 替换为 shift-add 操作。例如 K=2 时 √2 ≈ 2⁰ + 2⁻¹，实现为 shift 加 add。
    3. AE 输入：量化后的 weight code Q_W、激活 code X、控制信号（n₁, chk_even）。Decoder（组合逻辑）生成 Approx 模块使能信号和 Shift 模块移位位数。最终输出为 adder tree 累加的 partial sum。
    4. 2-term SDE 误差 ≤ 0.058，3-term SDE 误差 ≤ 0.0024。
  - 实验比较：
    - BRECQ AE：8/8-bit 非对称线性量化，需 INT SMUL（乘法器），面积 95.8 µm²，功耗 6.28 µW
    - AdaLog AE：8-bit weight（线性）+ 4-bit activation（对数），需 LUT + 乘法器 + 移位器，面积 76.2 µm²，功耗 5.56 µW
    - LogART AE（w/ HAF）：4-bit weight（DLog）+ 8-bit activation（线性），纯 shift-add 无乘法器，面积 53.2 µm²，功耗 3.45 µW —— 相比 BRECQ 面积减少 44.5%、功耗减少 45.1%，相比 AdaLog 面积减少 30.2%、功耗减少 38.0%
  - HAF accuracy impact: <0.2% accuracy degradation on vision models, <0.2 PPL on LLMs vs ideal LogART

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - Synopsys Design Compiler（商用 EDA 工具，论文未提供链接；官网 https://www.synopsys.com/implementation-and-signoff/rtl-synthesis-test/design-compiler-nxt.html）
  - 28nm UMC 工艺库，250 MHz，0.9V 供电
  - AE 设计综合：论文使用 SystemVerilog/VHDL RTL 描述各 AE 变体（线性/Log2/Log√2/AdaLog/LogART），通过 Design Compiler 综合得到面积和功耗数据

- 模拟器模拟什么的性能，修改了什么。
  - 模拟 AE（算术单元）在固定 4-bit weight × 8-bit activation 配置下的面积（µm²）和功耗（µW）
  - 对比了 5 种 PTQ 方法对应的 AE 设计（Figure 4）：
    (a) Linear AE（INT SMUL 乘法器）
    (b) Log2 AE（纯移位器，无乘法器）
    (c) Log√2 AE（shift-add，通过 √2 ≈ 2⁰+2⁻¹ 近似）
    (d) AdaLog AE（LUT + 乘法器 + 移位器）
    (e) LogART AE（Decoder + Approx 模块 + Shift 模块 + adder tree，全 shift-add 实现）
  - 论文修改/新增：LogART AE 为全新设计，其 Decoder 由简单组合逻辑实现，Approx 模块执行 SDE 展开的 shift-add，Shift 模块执行基-2 部分移位

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - AE RTL 源码未明确说明是否在开源仓库中包含；主代码仓库 https://github.com/logart-lab/logart
  - Design Compiler 综合流程（通用）：
    1. 输入：RTL（Verilog/VHDL）描述的 AE 数据路径，包含 weight 解码器、移位器、adder tree
    2. 设置约束：clk 250 MHz，0.9V 供电，28nm UMC 标准单元库
    3. 综合：DC 将 RTL 映射到标准单元门级网表
    4. 输出：面积报告（µm²）、功耗报告（µW，基于开关活动性）、时序报告
  - LogART AE 的数据路径（Figure 4(e)）：
    - 输入端口：Q_W（4-bit 量化 weight code）、X（8-bit 激活）、n₁（base 配置）、chk_even（奇偶校验）
    - Decoder：组合逻辑，判断每个元素属于 base-2 还是 base-√2 区域，生成 `enable_approx` 和 `shift_amount`
    - Approx 模块：当 enable_approx=1 时，执行 SDE 展开的移位加法（如 K=2: 不移 + 右移1位求和）
    - Shift 模块：对基-2 元素，执行 X << shift_amount（即乘以 2 的幂次）
    - Adder tree：累加所有 partial sum
  - 论文 Table 6 给出的结果：LogART AE 在保持 SOTA 准确率的同时实现最小面积和功耗
