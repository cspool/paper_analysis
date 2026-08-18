## Multi-Layer Execution（MLX，多层折叠执行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Multi-Layer Execution（MLX）是论文提出的空间数据流执行抽象：把深层结构化算子（FFT、蝴蝶稀疏矩阵乘 BSMM、block MM、sliding-window attention 等）在"时间上折叠（fold）"到紧凑固定 PE 阵列上执行——因为每阶段有有界的阵列驻留足迹，只有部分阶段同时驻留阵列，其余按依赖顺序时间复用。MLX 解耦逻辑阶段深度与物理阵列大小，使深层流水通过折叠执行在紧凑阵列上成为可能（论文 IV-C）。它不是苹果的 MLX 机器学习框架，而是 ISCA 2026（ICT CAS）的 "Multi-Layer Execution" 架构名。核心组成：(1) 算子抽象层——Chunked FFT 与 hierarchical BSMM 可表达为层对齐、前向-only 依赖的阶段序列（CDC 层），见"Closed Dependency Components (CDC)"条目；(2) 硬件实现层——skip-hop 网格 NoC（有界跳转路由）+ tagged-block 指令（层粒度调度）+ 解耦 compute/transfer 流水（跨层重叠），见对应条目。MLX 通过"层内确定性 + 跨层弹性"的混合调度（编译器固定层内静态指令序列，硬件只做 tag 级跨层协调）避免完全动态调度的大状态与完全静态调度的全局时序推理。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
MLX 架构运转流程（Fig.9 全设计，4×4 网格、32-way SIMD、12nm @1GHz、1 TOp/s FP16）：RISC-V host controller 发紧凑命令 → 每 PE 用 tagged block（LD 头/COMP 中/XFER 尾的固定布局 + loop trip count）在 active-layer window 内同时推进多个折叠层——一层正在加载输入、一层正在计算、一层正在转发结果 → 跨层传输经 skip-hop 网格按蝴蝶 stride（±2/±4/±8）1-2 跳路由 → 部分结果留在阵列内连续跨层流水（不往返全局内存）→ 结果写回。dense MM 也可折叠（每 PE 算 8×8 SIMD tile、psum 沿前向 operand 传播、tile 序列作为 MLX 层重叠，适合小 K 或 partial tile 场景）。效果：BSMM/FFT 计算利用率约 90%，kernel launch 开销 17%→<12%；端到端（稀疏 Llama2-7B vs 稠密 Xavier）3.2× 加速/3.1× 能量节省；相对先验稀疏加速器最多 5.8×；4×4→8×8 网格近线性扩展（SIMD 3.9×、mesh 3.6×、联合 14×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL 综合于 12nm @1GHz（Synopsys DC），从真实流片通用 dataflow 设计按 profiling（FFT/BSMM/dense kernel）裁剪（SIMD 32→8、去掉 vector shuffle/除法/高精度浮点），reduced 设计 256 GOp/s、0.772 mm²/433.8 mW（占原芯片 10% 面积/8% 功率）；软件部署用 RISC-V host + LLVM-based C 编译器或 dataflow 式汇编经轻量 "spatial assembler" 编译成 header 配置。设计参数原理：SIMD≥8（蝴蝶稀疏有效）、4×4 mesh + 每 PE 32 条指令满足覆盖条件（B_T·C ≥ T_load+T_xfer）、FP16 最小稳定精度（支持 8192 点 FFT、4096 个 twiddle）、超越函数单元为 1/4 SIMD 宽。使用：程序员写 dataflow 式汇编或 C 经编译器生成 tagged-block 配置；评估用 SimICT 周期精确模拟器（reduced 设计）+ 流片实测（全设计）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
