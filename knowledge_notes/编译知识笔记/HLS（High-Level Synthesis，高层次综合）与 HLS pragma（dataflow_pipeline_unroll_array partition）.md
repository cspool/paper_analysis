## HLS（High-Level Synthesis，高层次综合）与 HLS pragma（dataflow/pipeline/unroll/array partition）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HLS 是把 C/C++（或 SystemC 等）算法描述自动翻译为 RTL（Verilog/VHDL）的编译技术，是 FPGA 数据流加速器开发的主流入口；CODO 的最终产物即 HLS C++ kernel，交给 AMD Vitis HLS 2023.2（配合 Vivado 2023.2）综合出硬件。HLS 用 pragma/directive 控制调度与资源映射，CODO 流程中自动插入的核心 pragma：`#pragma HLS dataflow`（任务级流水）、`#pragma HLS pipeline`（循环/函数流水）、`#pragma HLS unroll`（循环展开）与 array partitioning（数组分区，把 BRAM 数组拆成多 bank 供并行访问）。
- dataflow pragma 的语义与约束（Web 证据：Xilinx Vitis-Tutorials Cholesky module4_dataflow、UG902）：在顺序函数/循环间插入通道使消费者在生产者完成前开始执行；数组通道默认按访问模式选实现（顺序访问→深度 1 FIFO，非顺序→ping-pong RAM，可用 config_dataflow -default_channel fifo|pipo -fifo_depth N 覆盖），标量/指针/函数返回参数一律 FIFO。硬约束：每个通道必须 single-producer-single-consumer；违例（单产多消、多产单消、任务旁路、任务间反馈、条件执行、多出口循环）使 HLS 跳过 dataflow 优化——这正是 CODO 粗粒度违例消除的动机。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 CODO 中 HLS 是后端而非前端：CODO 在 MLIR 上完成全部优化后，经扩展 HLS dialect（基于 HIDA 版本，用专用操作表达 dataflow pragma 与 array partition 指令）降级输出带 pragma 的 HLS C++ kernel + host code，再交 Vitis HLS 综合。pragma 的插入由自动调度 DSE 决定（PA/UP/DP 三阶段为每个循环选 pipeline/unroll/array partition 组合），不是用户手写。
- 运转流程例（ResNet-18 卷积层）：CODO 在与 FIFO 无关的最内层安全循环插 unroll + array partition（把 3×3×3 权重/输入窗口拆成多 bank 并行读），pipeline 放在合适循环层——对比 Allo 把 pipeline 放最外层导致所有内层全展开、而数组未分区时 27 个元素只能从双口 BRAM 顺序读、每次读产生 14-cycle 延迟的负优化。
- HLS 的局限（论文动机）：Vitis HLS 只报告粗粒度违例、不自动改写代码；细粒度违例（读写顺序/计数不一致）综合阶段根本不报错，cosim 才能发现死锁且大模型需数天到数周。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 商用实现：AMD Vitis HLS、Intel HLS Compiler、Siemens Catapult 等；用法 = 写 C/C++ + 插 pragma → 综合出 RTL + 时序/资源报告。CODO 的定位是把"重构代码与写 pragma"自动化：用户只给 C/C++ kernel（Polygeist 路径）或 PyTorch 模型（Torch-MLIR 路径），codo-opt 一条命令完成违例消除、缓冲选择、reuse buffer、片外传输与调度，输出可直接综合的 HLS 代码。复现依赖 Vitis HLS 2023.2（论文明确更低版本结果不一致）。

Graph.hls 的 HLS 视角（ISCA'26）：把 HLS 当作代码生成**目标**而非编程方式——GH-Architect 从 DSL/IR 直接产出完整 Vitis HLS 工程（HLS C++ kernel + host + Makefile + system.cfg SLR/HBM 绑定），交给 Vitis 2024.1 综合（约 4–6 小时/设计点），综合前正确性由 GH-Scope IR 级验证保障。论文强调 HLS 验证层两大缺口：(1) C-Sim 顺序执行模型无法模拟图加速器的并发 pipeline（无法建模有限流深度、检测不到死锁），只能靠硬件仿真 Co-Sim（~50 分钟/迭代），定位一个数据依赖 bug（如 16-bit 距离在直径>65535 的图上溢出回绕）需多轮 Co-Sim 二分；(2) late-stage 跨 SLR 布线失败要到 >3 小时综合后才暴露。Graph.hls 以 IR 级模拟 + golden reference 对比绕开 HLS 仿真层（301.6× 于 C-Sim、最高 455,000× 于 Co-Sim）。

NeRArch-Sim 的 Catapult HLS 视角（ISCA'26，HLS 作为模块级 PPA 快速估计通道）：NeRArch-Sim 的模块化硬件加速器用 SystemC + Catapult HLS（Siemens）实现 20+ 按统一分类学的硬件模块，HLS 综合用于"快速 PPA 估计"（分钟级），与全 ASIC post-layout 流程（Synopsys Fusion Compiler + PrimePower）对比验证（表 VI）：17 个模块延迟完全一致、面积/功率相对误差 4.72%~9.33%（ICARUS Pos Encoding 6714/5200 µm²、NeuRex Systolic Array(32×32) 5.4×10⁵ µm²、CICERO NPU(24×24) 3.1×10⁵ µm² 等）。HLS 可配置参数（表 IV）：pipelining II、loop unrolling、array partitioning（HLS 原生支持）、任意整型/浮点/定点精度、CORDIC/分段线性 exp 实现、systolic 尺寸/PE 数/buffer 深度、并行因子、通道深度/流宽/握手协议。安装/复现：Hardware/ 目录 + Catapult 2024.1_2 + MatchLib + PDK，S0_scripts 下 run_hls.py/run_fc.py/run_pwr.py 逐模块产出 PPA.log/timing.log。HLS 产物可部署到 FPGA/ASIC，是 NeRArch-Sim 高可实现性（Tab. I：✓ High Implementability）的来源。

涉及论文标题：
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
