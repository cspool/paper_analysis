## NDP 编译流程（Operator Partition → Loop Tiling → DRAM Mapping → Instruction Generation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NDP 编译流程是 DRAM 近数据处理架构的编译管线四阶段：① operator partition——把权重矩阵分块，每块映射到特定内存单元（如 DRAM bank），同时决定由哪个 PU 处理；② loop tiling——优化遍历顺序最大化 SRAM 缓冲内数据复用；③ DRAM mapping——为每块分配具体行列地址（即优化数据布局），目标降低 row-buffer miss（切行需 precharge，代价高于同行连续访问）；④ instruction generation——生成指令用于离线性能仿真与在线控制硬件，指令进一步展开为 DRAM 命令与 PU 命令。代表工作：OptiPIM（ILP 搜布局）、ATiM（UPMEM autotuning）、UniNDP（统一抽象+仿真）。核心特征：PU 与内存单元紧耦合，数据访问/计算/通信性能都取决于数据布局——这与 GPU/CPU 编译器复用不成的原因。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
FlexQ-NDP 在此基础上针对低比特 FP 扩展各阶段：① partition——考虑 QGroup 对齐（否则跨 PU 的 QGroup 造成 padding 与指令流发散，guideline-1）；② loop tiling——tile 尺寸对齐缓冲并保证 K_Tile ≤ Data_Buf 避免内层循环缓冲 miss，两种循环序 Order1/Order2；③ DRAM mapping——改用 scale-value 交织布局（见该术语条目）；④ instruction generation——插入 scale 读取/dequant/写回指令，再经去量化隐藏重排。低比特 FP 引入的新数据类别：scale、组内部分和，使缓冲分配从 (data, result) 二元变为 (Val_Buf, Scale_A_Buf, Scale_W_Buf, Dequant_Buf) 四元组。端到端例子（LLaMA2-7B 解码 MVM 1×4096×4096，W4A4S8）：权重按输出通道划分到 1024 PU → 每 PU 负责的块按 Order1 做 K 维循环切块 → 交织布局映射权重到物理行列 → 生成"读 scale → 流式读 value → MAC → dequant → 写回"指令序列 → UniNDP 仿真得 cycle 延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FlexQ-NDP 复用 UniNDP 的 INT 导向编译框架，在其上加量化元数据 IR 包装、scale/partial-sum 缓冲建模与三个新 pass（交织布局、dequant 重排、DSE）。使用：输入 QConfig + GEMM 描述 + NDP 硬件描述 → 输出指令流 → 交给仿真器（离线）或 NDP 硬件（在线）执行。搜索策略见"设计空间探索 DSE"条目的 FlexQ-NDP 视角。限制：编译空间巨大（划分×缓冲×循环序×映射笛卡尔积），必须裁剪 + 代价模型加速。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
