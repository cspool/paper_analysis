## Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference（PLENA）（近似层次匹配：自研 PyTorch→PLENA-ISA 编译器 + BoTorch 自动化 DSE，非对开源编译框架的修改）

- 属于编译框架的实现是什么？实验比较什么？
  - 实现为 PLENA 软件栈中的 (1) 轻量 PyTorch-to-PLENA-ISA 编译器：解析模型配置文件中的元数据，映射到预定义的 PLENA 自定义 ISA 汇编模板（Transformer 计算高度重复且结构统一，故编译器刻意保持轻量）；自动代码生成覆盖 MHA/GQA/MLA、Dense/MoE 等模型变体，含 FlashAttention 融合计算模板；(2) 自动化 accuracy-aware DSE 流程：基于 BoTorch 的多目标贝叶斯优化（目标 f=(f_accuracy, f_latency, f_area)），拒绝采样丢弃不满足约束的候选（MLEN·KV_WIDTH≤MemBandwidth、MLEN mod BLEN=0、MLEN≥HLEN≥BLEN），联合搜索 BLEN/MLEN/VLEN/M_LOAD/V_LOAD/V_WRITE/ACT_WIDTH/KV_WIDTH/FP_SETTING（Table III），产出 Pareto 前沿设计点。
  - 实验比较：DSE 采样方法对比 BoTorch（主动学习多目标 BO）vs TPE vs Random sampling（LLAMA3.2-1B：9 seeds×50 trials；LLaMA-3-8B：5 seeds×50 trials），指标为 EAS（Empirical Attainment Surfaces）上的 perplexity-latency Pareto 前沿质量；编译产物有效性体现在 Table IV 的 4 个 Pareto 设计点（BLEN 8–32、MLEN 128–1024、perplexity 6.54–6.76、latency 0.116–0.174 s、area 23.64–203.4 mm²）。
- 硬件平台是什么，配置是什么。
  - 编译目标 = PLENA 加速器（7 nm、1 GHz，flattened systolic array）；DSE 的 latency 目标评估跑在自研 Rust transaction-level emulator 上（cycle 级，集成 Ramulator/DRAMSys）；面积目标来自 Synopsys Design Compiler + 7 nm OpenROAD 预测 PDK 综合；BoTorch 运行平台论文未明确说明（quantization 在 H100 上完成）。
- 开源编译框架是什么。修改了什么。
  - 非开源框架修改：编译器为自研（模型 config 解析 → ISA 汇编模板映射），论文未说明基于 MLIR/TVM 等开源编译框架；DSE 基于开源 BoTorch（https://botorch.org）多目标贝叶斯优化，未报告修改其本体。
- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  - 开源：整体"coming soon"（https://plena-cam.github.io/，"Code Coming Soon!"，arXiv:2509.09505），编译器/DSE 代码未发布，无法确认链接；BoTorch 开源。
  - 编译框架输入到输出全过程：输入 = 模型 config（如 LLaMA-3-8B：hidden 4096、32 heads、head_dim 128、GQA 8 kv_heads）+ 校准数据（WikiText-2 抽样，PTQ 用）→ 编译器读取 config 元数据，把模型逐层映射为预定义 ISA 汇编模板：FFN（weight–activation GEMM (BLEN,MLEN)×(MLEN,BLEN) + GELU 等向量指令）、attention（QK^T GEMM 用 Matrix SRAM 转置读、online softmax 用 vector/scalar 指令的 max/sum/exp/div、PV GEMM、KV append 前的旋转+MX 量化指令）→ 输出 32-bit PLENA 指令流（M/V/S/H/C 五类共 47 条），经 PCIe 从 CPU 动态分发到指令缓冲 → 指令驱动 PLENA 矩阵/向量/标量单元与 HBM 传输（H_LOAD_M/H_LOAD_V 预取）执行；DSE 在此工具链外层以 BoTorch 采样候选 (BLEN,MLEN,VLEN,M_LOAD,V_LOAD,V_WRITE,ACT_WIDTH,KV_WIDTH,FP_SETTING)，对每候选跑量化（H100）+ transactional emulator（latency）+ DC 综合模型（area），拒绝采样剔除非法点后更新 GP 代理模型，迭代产出 perplexity-latency-area 三目标 Pareto 前沿（50 trials×5–9 seeds）。
  - 效果：BoTorch 显著优于 Random sampling 与 TPE 的 Pareto 前沿质量；co-design 选出最终系统配置 BLEN=32、MLEN=2048、VLEN=2048、W/A/KV=4/4/4 用于系统级评测。
