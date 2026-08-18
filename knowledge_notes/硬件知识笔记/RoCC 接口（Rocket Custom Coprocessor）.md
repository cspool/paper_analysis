## RoCC 接口（Rocket Custom Coprocessor）

术语解释
UC Berkeley Rocket Chip 生成器提供的自定义协处理器接口，把定制加速器紧耦合到 RISC-V 核（Rocket/BOOM）而无需改核流水线；论文用它作为 L2-attached OCA 的调用接口（RoCC-like 指令）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RoCC 通过自定义指令 opcode（custom-0/1/2/3 空间）把指令卸载到加速器：核解码后经 ready/valid 命令接口把操作码与解码后的寄存器操作数发给协处理器，协处理器计算后经响应接口返回结果与目标寄存器号，由核写回寄存器文件；可选附带 TileLink 内存接口与 FPU 接口。它是研究导向的接口（约束少、易扩展），Chipyard 中 Hwacha（向量协处理器）、SHA3 等是参考实现。论文在 baseline 中用它构造"L2-attached OCA"：同一空间位置的加速器，但调用走 RoCC 式接口——**不可推测执行**（RoCC 命令非推测发出），从而与 ATX 的"可推测/乱序调用"形成唯一变量对比：ATX NCA No Pref. 较 L2 OCA 仍快 1.6×/1.4×/1.3×（SpMM/SDDDMM/GeMM），证明调用模型本身（而非位置或流引擎）就是主要收益来源。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
调用流程（Rocket 核 + RoCC）：取指/译码识别 custom opcode → 读取源寄存器、组装 RoCC Command（含 funct、rs1/rs2 值）→ 加速器执行（期间核可继续执行不依赖其结果的其他指令，但不能推测执行其副作用）→ RoCC Response 携带 rd 值与目标寄存器索引 → 核写回寄存器文件。与 ATX 对比：ATX 指令在 ROB 中乱序发出、可被 squash；RoCC 风格调用被当作有副作用的操作，必须到达提交点或经额外同步才能发出，任务启动串行化，且轮询完成状态需要 fence——这正是论文图 2 刻画的 OCA 三宗罪。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 Rocket Chip 的 SoC 配置里用 LazyRoCC 挂接加速器（例化 `RoCCCommand`/`RoCCResponse` 端口 + TileLink 从口），配合自定义指令编码与编译器 intrinsic。学术界常用 RoCC 验证加速器原型的软件接口；产业界 RISC-V 方向则倾向 CV-X-IF、SCAIE-V、CXU-LI 等更规范的扩展接口（2025 RISC-V 扩展逻辑接口讨论会结论）。使用局限：无推测执行、核-加速器耦合接口带宽有限，恰好是 ATX 要解决的问题。

LIPPEN 的 RoCC 视角（ISCA'26）：把 RoCC 用作**指针加密/认证密码加速器的紧耦合挂接点**——LIPPEN 的 PRINCEv2 加密引擎与 baseline 的 QARMA 认证引擎均通过 RoCC 接口接到 Rocket（顺序）与 BOOM（乱序）核，加速器通过紧耦合请求/响应队列与核通信，对 64-bit 指针做单周期加/解密（PTR_SEAL/PTR_UNSEAL）。RoCC 的 ready/valid 命令-响应语义恰好契合"指针在 load-use/返回关键路径上解引用前必须完成解密"的时序要求；硬件成本数据（Table VI）显示 RoCC 接口本身是 FPGA 频率下降的主要来源（Rocket 150→110 MHz），密码引擎只贡献边际开销（LIPPEN RoCC 1,034 LUT vs PAC 2,071 LUT，QARMA 数据通路更复杂）。

- LoRA 中的 RoCC 用法（ISCA'26）：作为 CGRA 的调用接口——后端生成的 CGRA calling function 包含自定义 RoCC 指令序列（load 配置与数据到 SPM、配置 CGRA、激活执行、写回主存），CPU 执行这些指令经 RoCC 接口发往 reservation station，reservation station 缓存并按依赖把指令分发到 load/store 控制器、CGRA 控制器等；实现"CPU 管理数据 + 协处理器加速"的 SoC 集成模式。

涉及论文标题：
- ATX: Accelerator Task Extensions
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
