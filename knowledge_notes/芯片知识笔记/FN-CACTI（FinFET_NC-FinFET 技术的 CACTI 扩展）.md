## FN-CACTI（FinFET/NC-FinFET 技术的 CACTI 扩展）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FN-CACTI（Ravipati 等，IEEE TVLSI 2022）是 CACTI 系列面向 FinFET 与负电容（NC-FinFET）技术的扩展：在经典 CACTI 的 SRAM/cache 面积-功耗-延迟模型上加入 FinFET 器件的物理（鳍高/鳍数、工作电压、温度、工艺角）与 NC-FinFET 的铁电/负电容层参数化，供先进节点片上存储结构（SRAM 阵列、cache、scratchpad）的早期设计空间探索。论文中被 GenZA 用来建模 PE 内 128 kB SRAM scratchpad 与全局 transpose buffer 的面积与功耗（Table III/Table VII 的 SRAM 项）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 GenZA 芯片设计流程中的角色：PE 的算术逻辑（KO 阶段、乘法器、模加器、crossbar）用 Verilog RTL + ASAP 7nm 综合得到面积/功耗（如 Table VII：32 个 64-bit 乘法器 31k µm²/25 mW、768-bit KO 级 17k µm²/21 mW），而 SRAM 类存储（每 PE 128 kB scratchpad 51k µm²/8 mW）与全局 transpose buffer（0.9 mm²/3.1 W）由 FN-CACTI 按 7nm 级 FinFET 参数建模输出。两者相加得整 PE（166k µm²/164 mW）与整芯片（58.5 mm²/64.1 W，含 2 HBM2e PHY 29.8 mm²/31.7 W）的面积功耗分解。FN-CACTI 输出作为面积效率指标（ATP、性能/面积）的分母，直接支撑"统一架构 vs 专用单元"的比较。
- 流程例子：输入（容量 128 kB、bank/端口/工艺 FinFET 参数、目标频率）→ FN-CACTI 内部对字线/位线/解码器/传感器按 FinFET 物理建模 → 输出面积、读/写功耗、访问延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为 CACTI 的建模工具（集成 FinFET/NC-FinFET 器件模型）使用，输入 SRAM 结构参数与工艺参数输出面积/功耗/延迟；常与逻辑综合（如 ASAP 7nm + Design Compiler/OpenROAD）配合覆盖"逻辑 + 存储"的完整片上面积功耗评估。使用：加速器/微架构论文中对 SRAM scratchpad、cache、buffer 的建模（GenZA 用其建 scratchpad 与 transpose buffer）；论文未给出 FN-CACTI 的公开下载链接（References [61] 指向论文本身），联网未确认独立仓库。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
