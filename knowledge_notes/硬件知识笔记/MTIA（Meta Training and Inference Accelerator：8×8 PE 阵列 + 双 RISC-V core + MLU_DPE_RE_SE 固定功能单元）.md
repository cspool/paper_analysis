## MTIA（Meta Training and Inference Accelerator：8×8 PE 阵列 + 双 RISC-V core + MLU/DPE/RE/SE 固定功能单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MTIA 是 Meta 自研的面向推荐系统（DLRM）训练与推理的定制 AI 加速器。第一代（ISCA 2023，Firoozshahian 等）：TSMC 7nm、800MHz、102.4 TOPS INT8 / 51.2 TFLOPS FP16、25W TDP；2D 8×8 网格共 64 个 Processing Element（PE）经片上 mesh/NoC 互联，配片上 SRAM（每 PE 128KB 本地 SRAM + 128MB 共享片上 SRAM）与 LPDDR5 内存控制器。每个 PE 含两个 RISC-V 核（标量核 Andes AX25-V100 + 512-bit 向量核 NX27V，Meta 是 Andes 最大客户之一）+ 固定功能单元：MLU（Memory Layout Unit，数据搬运/布局转换）、DPE（Dot Product Engine，每周期 1024 INT8 MAC 或 512 BF16/FP16 MAC）、RE（Reduction Engine，矩阵乘累加、支持跨 PE 归约）、SE（SIMD Engine，量化/激活/查找表）、FI/CP（Fabric Interface / Command Processor）。MTIA 2i（Coburn 等，ISCA 2025）保持 8×8 PE 阵列 + 每 PE 双 RISC-V core + 四引擎 + CP。MTIA v3 为论文评测中的最新代，内存带宽/算子覆盖更强（MapId/MBDT 在 v3 上 latency 0.035-0.174ms vs v2i 0.399-8.090ms）。Web 来源显示 Meta 已规划 MTIA 300/400/450/500（与 Broadcom 合作，chiplet 设计、新低精度格式 MX8/MX4），并有面向 LLM 训练的 RISC-V 原型（>1000 核、200+ TFLOPS 传闻），论文标注存在歧义（"MTIA v3"既可指评测用最新代也关联 MTIA 300）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 作为异构加速器 fleet 中的"私有架构"成员运转（KernelEvolve 场景）：模型 kernel 经 Triton-MTIA 编译成 RISC-V 二进制在 PE 阵列执行——命令处理器（CP）调度，RISC-V 核做控制流与标量操作，DPE 做矩阵点积，RE 跨 PE 归约，SE 做激活/查找表，MLU 做布局转换。例子（MapId kernel 在 MTIA v2i 执行）：输入 categorical ID 张量 → 每个 PE 的 RISC-V 核按 BLOCK_SIZE 分块，in-register 20 步二分查找（编译期 unroll 为直通代码）→ SE/SFU 处理比较，结果经 coalesced load/store 写回。硬件异构性（相对 GPU）：自定义片上 SRAM 子系统、不同片上/片外带宽 profile、无 NVIDIA 式大 L2/AMD 式 Infinity Cache；kernel 需用 MTIA 专属原语（SFU LUT、cross-PE broadcast/reduction、dual-core 流水）才能到最优性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：MTIA 作为服务端芯片部署于数据中心（rack 集成，图 1），程序化接口经 Triton-MTIA（triton_mtia，`from triton_mtia.python.mtia.eager import mtia_triton_launcher; mtia_triton_launcher.init()` 为 MTIA 执行必需）与 PyTorch ATen 算子层暴露；KernelEvolve 的 meta_kernel_mtia_interpreter（Bento notebook 环境）捆绑 MTIA runtime + 编译器 + MTIA Insight profiler，经 Conveyor CD 每日重建。MTIA v2i 原生缺 clamp.out/gather.out/sort.values_stable/all.all_out/_unique2/unique_consecutive 等 ATen 算子（v3 覆盖更好但仍有缺口），缺 kernel 时 PyTorch 回退 CPU（跨节点开销），这正是 KernelEvolve 自动化 kernel 生成的动机。评测：MapId MTIA v2i 最高 4.07×、v3 最高 1.36×；MBDT v2i 2.94-9.25×、v3 2.31-3.09×；RMSNorm 2D backward 17×。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta
