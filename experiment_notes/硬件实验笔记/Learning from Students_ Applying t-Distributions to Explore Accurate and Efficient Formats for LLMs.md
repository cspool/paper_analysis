## Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs

- 属于硬件架构的实现是什么？实验比较什么？
  - **多数据类型 MAC 单元硬件设计**：为 11 种 4-bit 数据类型（INT4, INT5, E2M1-I, E2M1-B, E2M1, E2M1+SR, E2M1+SP, E3M0, APoT4, APoT4+SP）设计了对应的 MAC（乘加）单元。每个 MAC 包含 multiplier 和 accumulator，accumulator 位宽按无损累加 256 项 dot-product 要求设定（不同格式动态范围不同，所需 accumulator 位宽也不同）。
  - 实验比较所有数据类型的 MAC 面积（µm²）和功耗（µW），以及估算的系统级芯片面积开销（假设 MAC 占芯片 10%、存储占 60%）。
  - 关键结果：
    - INT4 面积最小（160.7 µm²）、功耗最低（48.5 µW），但精度最低
    - 标准 E2M1 系统开销仅 0.6%（vs INT4），但平均精度损失降低 7.34%
    - E2M1+SP 相比 INT4 仅增加 3.6% 系统面积，精度大幅提升
    - E2M1-I（Intel 变体）和 E2M1-B（bitsandbytes 变体）因动态范围过大（需 20/23 bit accumulator）属于严格劣化的 Pareto 点
  - 质量-效率 Pareto frontier：INT4 → E2M1（0.6% 开销）→ APoT4（中等）→ E2M1+SP（最高精度，3.6% 开销）

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - Synopsys Design Compiler（商用 EDA 综合工具），论文未提供链接；官网 https://www.synopsys.com/implementation-and-signoff/rtl-synthesis-test/design-compiler-nxt.html
  - TSMC 28nm 工艺库
  - 使用 SystemVerilog 描述各数据类型的 MAC 单元 RTL

- 模拟器模拟什么的性能，修改了什么。
  - 模拟各数据类型 MAC 单元的面积（µm²）和功耗（µW）。每个 MAC = 4-bit multiplier + N-bit accumulator。
  - 各格式 accumulator 位宽：INT4=16bit, E2M1=17bit, E2M1+SR=18bit, E2M1+SP=19bit, E2M1-I=20bit, E2M1-B=23bit, E3M0=22bit, APoT4=16bit, APoT4+SP=16bit, INT5=18bit
  - 低比特下 accumulator 面积可能超过 multiplier（如 E2M1 accumulator 比 multiplier 大 13.8%），与高精度下 multiplier 面积平方增长占主导不同
  - 系统级开销估算：假设 MAC 占芯片 10%、存储 60%，则系统开销 = (MAC 面积比 - 1) × 10% × (1/13%)

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 论文未明确说明 MAC 单元 RTL 源码是否包含在开源仓库中；主代码仓库 https://github.com/cornell-zhang/llm-datatypes
  - Design Compiler 综合流程：
    1. **输入**：SystemVerilog RTL 描述的 4-bit MAC 单元（每种数据类型独立的 multiplier + accumulator 数据路径）
    2. **约束设置**：TSMC 28nm 标准单元库，target clock 约束
    3. **综合**：DC 将 RTL 映射到标准单元门级网表，执行逻辑优化
    4. **输出**：面积报告（µm²，含 multiplier 面积和 accumulator 面积）、功耗报告（µW，基于开关活动性估计）、系统级开销外推（基于 MAC/存储面积占比假设）
  - 硬件评估目的：建立质量-效率 Pareto 曲线，指导下一代 DNN 加速器在 4-bit 精度下的数据类型选择
