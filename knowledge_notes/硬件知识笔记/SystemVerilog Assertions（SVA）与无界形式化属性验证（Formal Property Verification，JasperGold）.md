## SystemVerilog Assertions（SVA）与无界形式化属性验证（Formal Property Verification，JasperGold）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SVA 是 SystemVerilog（IEEE Std 1800）内嵌的断言语言，用 concurrent assertion（如 assert property、cover property）表达跨时钟周期的时序属性；形式化属性验证（FPV）用 BDD/SAT/其他引擎把属性对所有可达状态做无界证明（不需要测试向量），反例即设计 bug。JasperGold 是 Cadence 的商用 FPV 工具（QED 引用 [1] JasperGold Platform and Formal Property Verification App）。SVA 三类语句在 QED 中的用法：cover 证明谓词前置条件可达、assert 证明"A 蕴含 B"（A|->B，对所有可达状态成立）、assume 约束模块输入合法范围。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
QED 把决策树谓词逐条翻译成 SVA（图 11），直接引用 BOOMv3 LSQ 的 RTL 信号：(a) cover(load_matrix[i][j] & ldq[j].succeeded & !ldq[i].succeeded) 证明"load j 乱序先完成"可达；(b) cover(... & ldq[j].observed ...) 证明"乱序 load 已匹配 invalidation"可达（对应谓词 Q2）；(c) assert(load_matrix[i][j] & commit_load_idx==i & ooo_load_matrix[i][j] |-> (ldq[i].addr!=ldq[j].addr) || !ldq[j].observed)——提交 load i 时若更年轻 load j 曾乱序执行过，则要么地址不同、要么 j 未被 observed（observed 的年轻 load 在 i 执行时应被 squash）；(d) assume(uop.valid |-> is_load XOR is_store) 约束每个 uop 只能是一个 load 或一个 store。JasperGold 证明 (c) 对所有可达状态成立即谓词通过；反例直接定位到具体 RTL 信号（如 stq[j].in_flight 时序、ldq[j].observed 过早置位）。模块输入假设（assume）常由产生该输入的另一模块的输出断言证明；BOOMv3 文档过时需逆向工程推断块级接口，或用 LLM 把自然语言接口定义转 SVA（[40][54][69][72]）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：设计者/验证工程师写 SVA 属性与约束 → 综合工具读入 RTL netlist → FPV 引擎无界证明（QED 用 JasperGold；开源替代如 Yosys-SMT/SymbiYosys，论文未用）。backward-slicing 剪掉 83% 无关输入（91% bit 自动 assume 为 0，如 uop.is_br=0——is_br 只为分支设置，LSQ 中无用）。QED 对 BOOMv3 LSQ 的 227 个 RVWMO 谓词逐个 proof，度量 full proof time（墙钟）、full proof depth（JasperGold 工作量）、memory consumed（证明引擎内存）；LSQ 尺寸翻倍 → 时间约×10（2^n vs 10^n）、深度近似线性、内存近似二次。电路级综合优化（如 LSQ 合成为 CAM）对任何 RTL 验证（含 QED）不可见，需单独验证。

æSIP（ISCA 2026）给出了 SVA 的另一种核心用法：不用于"证明性质"，而用于"表达 External Don't Care（EDC）约束注入 netlist 以支撑门级裁剪"。其 SVA generator 把静态分析重写汇编得到的约束自动转成 SystemVerilog assume 语句（三类）：ISA 级（decoder 只允许重写程序出现的 opcode，如 `assume ((i_rdata[1:0] != 2'b11) || (((i_rdata[31:0] & 32'hfe00707f) == OP_ADD) || (... == OP_MUL)))`）、数据级（立即数/移位量受限，如 `assume ((opcode == SRAI) |-> (shamt inside {1, 2}))`）、时序级（访存子系统时序，如 `assume (dcache_miss |-> ##[1:5] dcache_response)`）。这些 assume 声明"某些输入模式永不出现"，把综合问题从完全指定布尔函数松弛为布尔关系；abc 的 scorr/dsec 在 assume 约束下用 k-induction 证明节点等价/冗余后即可安全裁剪。注意与 QED 用法的区别：QED 用 SVA 对 RTL 做无界属性证明（assert/cover），æSIP 用 SVA 的 assume 语义做综合约束（EDA 约束注入），不依赖 FPV 引擎的无界证明。

涉及论文标题：
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
