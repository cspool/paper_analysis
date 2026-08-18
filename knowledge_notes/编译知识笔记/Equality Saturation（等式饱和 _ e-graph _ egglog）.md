## Equality Saturation（等式饱和 / e-graph / egglog）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Equality Saturation（等式饱和）是程序重写/编译器优化领域的技术，核心数据结构是 e-graph（equivalence graph）：由 e-node（操作符节点）与 e-class（等价节点集合）组成，紧凑编码大量语义等价的表达式。工作流两阶段：①饱和（saturation）——从初始程序表示出发反复应用全部重写规则直到不动点（e-class 合并等价节点，规则可递归、交叉应用，形成稠密等价空间）；②提取（extraction）——用用户定义成本模型从 e-graph 选择一组 e-node 组合，产出语义等价的优化程序。关键优势：规避 phase-ordering 问题（传统编译器优化效果依赖变换应用顺序；等式饱和同时应用所有规则，顺序无关）。开源实现：egg（Willsey et al. POPL 2021，https://github.com/egraphs-good/egg）、egglog（Zhang et al. PLDI 2023，https://github.com/egraphs-good/egglog，datalog 语义 + 等式饱和，æSIP 用 11.4.0）。应用领域：代码优化、technology mapping、电路 datapath 综合（E-Syn/E-Morphic）、浮点精度优化（Herbie）、算术推理、HLS 超优化（SEER）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
æSIP 的分治等式饱和（§IV-B，Fig.6）——直接全程序饱和不可行（规则非局部递归扩张、e-graph 复杂化使提取成本主导 [82]；既有工作要么在高层次 IR 粗粒度饱和 [20][84]、要么小局部顺序应用 [61]）：
```
① CFG 分析把程序切成 basic block 集合
② 每 block 局部饱和：pseudo-root e-class 锚定该 block 全部表达式
   （如 mul→callmul 序列、mulh→Karatsuba mul 序列、blt→bge+jal、lh→lw+slli+srai
    51 条规则同时应用，e-class 合并等价节点）
   orphan e-class（不参与本 block 主表达式树但被其他 block 消费的值）显式链回
   block pseudo-root —— 防止提取器把跨 block 依赖误判为死代码
③ 局部饱和子图并入单一全局 e-graph：global root 连接各 block pseudo-root，
   保留跨 block 等价选择，避免整体饱和组合爆炸
④ ILP 全局提取（见"硬件感知 ILP 全局提取"条目）：27 个 λ 值各产出一个重写变体
⑤ 后处理：重建表达式树、后序发射汇编、线性扫描寄存器分配 → 可直接汇编的 .s 变体
```
效果：extraction 0.3-78.4s（中位 21.9s）/benchmark；全局 monolithic saturation+extraction 在控制 e-node/e-class 规模下仍 1 小时超时；分治 + 隐式块级并行（各 block 共享全局指令类型变量）把全局提取从小时级降到秒级。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：egg/egglog（Rust 库）；egglog 用 datalog 风格规则语言定义 e-graph 转换（saturate 循环）。使用：定义目标语言 IR（指令/操作符）→ 定义重写规则（lhs→rhs，æSIP 中由程序合成阶段产出）→ run 饱和 → 提取（成本模型或 ILP）。æSIP 的提取目标不是最小延迟而是"硬件感知"：最小化 distinct 指令类型数（配合硬件权重 w_o），故用 ILP 而非贪心。运行时：saturation 1.2-32.4s/benchmark。生态：egg 系在编译器（DialEgg MLIR 优化、SEER HLS 超优化）、逻辑综合（E-Syn、E-Morphic 结构探索）、算术优化（Herbie）广泛应用，æSIP 展示其在新方向（ISA 子集裁剪/ASIP 生成）的可用性。

涉及论文标题：
- æSIP μArch-aware ASIP-ISA Co-Design via Program Synthesis, Equality Saturation, and External Don't Cares
