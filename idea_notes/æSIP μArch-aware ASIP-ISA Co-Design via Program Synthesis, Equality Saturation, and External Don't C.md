## æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares

- baseline方法是什么？
  - Baseline 是 SOTA 的 Reduced-ISA Set Processor (RISP) 设计方法学，以 PDAG [11]（及 Bespoke [22]、RISSP [58]、FSYN [34]）为代表：对目标应用做软件 profiling 收集指令使用信息，翻译成 ISA 级约束（只保留用到的 opcode 集合），再用 LTL model checking 从 baseline 通用处理器（Ibex RV32IM 等）netlist 中剪除未使用指令对应的硬件门。其痛点：①编译器只为性能优化生成代码、不会为减少指令使用而改写程序——profiling 只能看到"编译器给什么就用什么"，即使 mul 只被调用几次也必须保留整个乘法器硬件（bitcnts 的 PDAG 面积 ~1.16×10^5 µm²、功耗 ~4.0 mW 无法再降）；②约束粗粒度且微架构无关——只做 ISA 级 opcode 约束，忽略应用特定的数据模式（操作数实际宽度、立即数取值子集）与时序特性（cache miss 行为），桶形移位器、访存状态机等未被充分利用的硬件无从裁剪；③专一化 vs 泛化的权衡未解决——per-application 极端专一化面积收益大但 NRE 成本高、软件升级受限，共享 ASIP 又必须取各程序原始 ISA 的并集（workload 增加则并集膨胀、专一化收益被侵蚀）。
  - baseline 全栈执行例子（以 MiBench bitcnts 在 Ibex RV32IM 上为例）：
    ```
    算法pipeline层：论文未明确说明（无算法模型；目标是嵌入式 benchmark 程序 bitcnts 的执行）；
    系统框架层：论文未明确说明（无 serving 框架）；
    编译框架层：riscv32-unknown-elf-gcc -O3 生成汇编，编译器以性能为目标，mul/mulh 等复杂指令仍出现在汇编中；
    kernel调度层：论文未明确说明（无 GPU kernel；指令在 Ibex 2-stage in-order pipeline 顺序执行）；
    硬件架构层：profiling 统计指令使用 → 生成 ISA 约束（bitcnts 用到 mul 则乘法器必须保留）→ LTL model checking
               从 Ibex netlist 剪除未使用指令对应的门 → PDAG ASIP：面积 ~1.16×10^5 µm²（SKY130）、功耗 ~4.0 mW，
               少量 mul 使用使乘法器无法裁剪；只做 ISA 级约束，shamt 子集/cache 时序等无约束可剪。
    ```
- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 方法：æSIP 把软件当作可变的（通过可验证重写），实现 ASIP-ISA 硬件-软件协同设计：端到端输入目标程序 + baseline 处理器 + 时延约束，输出优化 ASIP 与可验证等价的重写程序。(1) 程序合成发现重写规则（Greenthumb 三算法 + Claude Opus 4.5 LLM 神经符号搜索，Z3 验证，51 条规则覆盖 26 指令）：mulh/mul/div 等复杂指令可改写为 add/shift/分支序列（Karatsuba/convolution 风格）——直接对应"编译器不为减少指令使用而改写"的缺陷，把"低使用高面积"的乘法/除法单元从 ISA 层面消除（bitcnts 类 workload 乘法器可全删，distinct 指令平均降 31.8%、Mul/Div 全消除）；(2) 分治 equality saturation + 硬件感知 ILP 全局提取（basic-block 局部饱和 + pseudo-root/orphan 处理 + Gurobi 按硬件权重 w_o 联合最小化指令类型集与总数，27 个 λ 扫面积-时延权衡）——对应"只能看到单一 profiling 程序、无权衡空间"的缺陷，同一程序产出 Pareto 前沿变体族（bitcnts -21.9% 面积 + 0.79× 时延，dijkstra -14.4% 面积 + 1.02× 时延）；(3) don't care 微架构感知裁剪（ISA + 数据 + 时序三级 EDC → SVA assume 注入 → abc scorr/dsec 在 k-induction 下证明并裁剪）——对应"约束粗粒度、微架构无关"的缺陷，在 ISA 级之外用 shamt∈{1,2}、dcache miss 5 cycle 内响应等约束剪裁桶形移位器与访存状态机（保留复杂指令的 dijkstra/rijndael 额外收益最大）；(4) ecosystem 级共享（ILP 联合把 num-chip 个 ASIP 分配并选重写变体，程序先改写趋同于公共 ISA 子集再取并集）——对应"专一化 vs 泛化、NRE 高"的缺陷，重写先收窄各程序 ISA 足迹再共享，num-chip=5 时 17.3% 面积降 + 11.9% 时延（逼近 per-application 的 22.4%/15.1%），且新程序可改写映射到既有 ASIP 子集实现泛化。
  - 论文方法全栈执行例子（同一 bitcnts 程序）：
    ```
    算法pipeline层：论文未明确说明（无算法模型；重写规则是 ISA 级指令等价变换，非算法层加速）；
    系统框架层：论文未明确说明（无 serving 框架）；
    编译框架层：输入自包含 RV32IM 汇编 → egglog 按 basic block 局部饱和（mul→callmul 序列、mulh→Karatsuba mul
               序列、blt→bge+jal 等 51 规则同时应用，规避 phase-ordering）→ Gurobi ILP 按硬件权重 w_o 全局提取
               （λ 扫 27 值）→ 后处理发射重写汇编 + 线性扫描寄存器分配 → 每 λ 一个变体（bitcnts distinct 指令
               平均降 31.8%、乘法器指令全消除），spike 验证与原程序等价；
    kernel调度层：论文未明确说明（无 GPU kernel；重写后 mul 变为数十条 add/shift/分支序列在 in-order pipeline
               顺序执行，cycle 增加，但简化 datapath 提高频率部分补偿——1.2× 时延约束下面积仍降 21.9%）；
    硬件架构层：对每个重写变体提取 ISA+数据+时序 EDC → SVA 注入 Ibex netlist → abc scorr/dsec k-induction 证明并
               裁剪等价/冗余节点 → ASIP：bitcnts 面积 -21.9%、功耗 -16.4%（SKY130，1.2× 时延约束）；无约束能量
               最优变体能量 -16%（频率提升 1.49× 缩短静态功耗时间）；ecosystem 级共享：5 个共享 ASIP 覆盖 22 个
               benchmark，17.3% 面积降 + 11.9% 时延，接近全 per-application 专一化。
    ```
