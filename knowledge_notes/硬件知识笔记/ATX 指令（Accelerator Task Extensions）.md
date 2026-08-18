## ATX 指令（Accelerator Task Extensions）

术语解释
ATX 论文提出的一组 ISA 指令 + 核流水线扩展，用于核推测/乱序地调用 NCA：从核的视角，一条 ATX 指令等价于一条"长延迟 load"，最终把结果送回核寄存器。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ATX 指令格式：(1) opcode 决定输入/输出操作数的数量与类型——如 `ATX V2V1`（2 个输入向量寄存器 + 1 个输出向量寄存器）、`ATX V1T2`（1 个输入向量 + 2 个输出 AMX tile 寄存器），不同 opcode 适配不同 NCA 任务；(2) 输入寄存器携带任务元数据：第一个输入包含 VAccId（任务类型标识），其余携带 UTE 生成访存地址所需的运行时常量（以及可选直传 NCA 的控制数据）；(3) 输出寄存器接收任务结果。指令提交三条件：任务完成、输出已写 PRF、指令在 ROB 头。ATX 指令是 NCA 无关、可复用的；错预测 squash 时 UTE 中断 NCA 任务，架构状态不受影响。核流水线为此新增：ATX Queue（16 项，跟踪 ROB 中 ATX 指令）、参与核 wake-up 的 ATX Reservation Stations、把指令发往 ATX Port 的调度逻辑；UTE 内部队列满时产生结构冒险阻止进一步发射。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
与普通指令的对比（论文图 6 ROB 快照）：ATX 指令**不需要**到达 ROB 头即可发出，只要输入（renamed）寄存器就绪就从 Reservation Station 发出；因此 ROB 中可同时有多条 ATX 指令处于不同阶段——Ins1 已完成未提交、Ins2 因依赖未就绪等在保留站、Ins3 已在 UTE 中执行、Ins4 正在发出——而 OCA 的 MMIO 调用必须串行地在 ROB 头执行且要 fence。提交仍与普通指令同序。论文还指出 ATX 未增加 PRF 写端口（与 INT/FP/LD/VEC 端口竞争写 PRF），且因一条 ATX 指令替代多条 load，PRF 压力反而下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用例（论文图 12 的 CSR 行求和任务）：`in_vregister = {VAccId, c11, c12, c21, c22, 0...}` 把任务类型与 4 个运行时常量打包进向量寄存器，然后 `ATXV1V1{in_vregister, out_vregister}` 触发 16 行稀疏矩阵行的求和任务；循环可 OpenMP 并行到多核。编译器支持：论文提供 C/C++ 小库定义 UTE 配置函数 + 内联汇编定义 ATX 指令；未来方向是编译器自动做"非 ATX → ATX"源到源变换（检测可加速段、识别流与依赖、切任务、生成调用），以及 NCA/ICA/OCA 自动分区。

涉及论文标题：
- ATX: Accelerator Task Extensions
