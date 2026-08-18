## Transaction-level 模拟器（cycle-approximate emulator + Ramulator/DRAMSys 多保真）

术语解释
介于解析模型与 RTL 仿真之间的多保真评估层：以 cycle 近似精度执行指令流并建模内存事务，用于加速器架构评估与 DSE。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PLENA 自研 Rust 事务级（transaction-level，cycle-approximate）模拟器：事件驱动执行编译器产出的 PLENA 机器码，以 cycle 粒度建模计算执行、指令调度与内存事务；集成 Ramulator（CMU-SAFARI 开源的 DRAM 时序/能耗模拟器）与 DRAMSys（TU Kaiserslautern 的 SystemC/TLM DRAM 系统级模拟器）提供 off-chip 时序/带宽/bank 级行为。PLENA 多保真框架共三层（图 10）：解析模拟器（8 ms，对比 RTL 综合：延迟误差 11.32%、面积 4.79%、功耗 23.81%）、事务级模拟器（4.3 min，延迟误差 4.17%，比 RTL 的 14 h 快约 200×）、RTL 仿真/综合（参考基准）——事务级被选为 DSE 的默认保真度。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
评估流程：输入 = (a) 编译器产出的 32-bit PLENA 指令流与张量布局、(b) 架构配置（BLEN/MLEN/VLEN、MX 精度、SRAM 容量、HBM 参数）、(c) workload token 序列（OSWorld-L 90k/8k、GSM8K 1.4k/0.2k、BFCL-W 114k/5k）→ 事件驱动逐 cycle 执行：decode → 矩阵单元 GEMM（systolic 数据流 + M_SUM 加法树）、向量/标量指令、HBM 事务（H_LOAD_M/H_LOAD_V 预取与写回，MX 数据布局）→ 内存事务转发 Ramulator/DRAMSys 计算 DRAM 时序 → 输出 TTFT（s）、TPS（tokens/s）、compute/memory active 时间、SA 利用率、带宽利用率（图 13 分阶段分解）。作用：定量分析内存-计算交互（内存带宽是长上下文推理首要瓶颈），并充当 DSE 中每个候选设计的 latency 评估器；与 RTL 对拍验证保证事务级抽象可信。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Rust 事件驱动执行引擎 + Ramulator/DRAMSys 作库集成（未修改二者本体）；验证 = 与自研 RTL 综合结果对拍（延迟与数值精度，Table II）。同类实践：LLMSimulator（SNU，C++ graph 执行 + Ramulator 2.0，支持 MHA/GQA/MQA/MLA/MoE 与 H100/B100/B200）、SCALE-Sim v3（Ramulator 集成）、Sieve（Ramulator 2.0 的 PIM cycle 仿真）——"自研架构模拟器 + Ramulator 内存模型"已是 LLM 加速器评估的主流组合；DRAMSys 常见于 ESL 级内存系统探索。使用：无法流片时对加速器做可信端到端 LLM 推理评估与多目标 DSE；精度上限由事务级抽象决定，关键机制（如扁平阵列映射、FlashAttention 调度）需与 RTL 校准；与商用平台（A100/H100/TPU）比较时按乘法器数与 HBM 配置对齐设备数。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference
