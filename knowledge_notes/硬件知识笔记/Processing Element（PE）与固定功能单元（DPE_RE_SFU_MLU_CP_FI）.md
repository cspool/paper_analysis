## Processing Element（PE）与固定功能单元（DPE/RE/SFU/MLU/CP/FI）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Processing Element（PE）是 MTIA 300 计算 chiplet 的基本计算单元，12×6 网格共 72 个。每 PE 含：2 个 64 B 宽 RISC-V 向量核（执行应用代码、向 Command Processor 发自定义指令，异步 dataflow 执行模型——kernel 生成固定功能单元指令序列、数据移动与计算按依赖就绪执行）、512 KB 软件管理 local memory（LS，拆成可控大小 Circular Buffers CB）、Memory Bridge（内部 NoC 连接各组件 + 中断控制器/机器定时器/debug-trace 外设），以及五个固定功能单元：(1) **Memory Layout Unit（MLU）**——transpose/reshape/slice/concat 布局变换；(2) **Dot Product Engine（DPE）**——GEMM，第一输入读入并缓存在 DPE、第二输入从 LS 流式进入与第一输入所有行做点积；双 32×64B×32 MAC tile，FP16/BF16 输入 FP32 输出共 7.82 TFLOPS/PE，支持 FP8（S1E4M3/S1E5M2）与 TF32；(3) **Reduction Engine（RE）**——存 DPE 中间结果、经专用归约网络做跨 PE 归约，可转发到下一 PE 或 SIMD 引擎；(4) **SIMD Engine（SFU）**——量化/逐元素/非线性，浮点 ALU + LUT 近似非线性；训练版去掉 INT8 加 FP8，SIMD 宽从 32 增到 128 元素/cycle（GEMM:SIMD 16:1 vs MTIA-2i 32:1）；支持 min/max/clamp/stochastic rounding 与硬件 radix sort（加速 embedding 反向）；(5) **Command Processor（CP）**——执行 RISC-V 核发来的自定义指令、调度与依赖检查、仲裁 LS 访问、提供 CB 抽象与生产者-消费者依赖跟踪；外加 **Fabric Interface（FI）**——DMA 引擎（PE 本地内存与片上/片外内存间搬运），包分片 + leaky-bucket 流量整形。FI/CP 增强：字节对齐 DMA（tensor 切片免软件布局开销）与硬件索引 DMA（按 LS 中的索引列表做 scatter/gather，用于 embedding 查表）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
一次 DLRM 训练 kernel 的 PE 内流程：RISC-V 核执行 kernel 代码 → CP 派发自定义指令到固定功能单元 → DPE 从 LS 流式读第二输入做 GEMM（第一输入常驻 DPE）→ 结果进 RE（跨 PE 归约网络做层间归约）→ SFU 做量化/非线性/radix-sort（embedding 反向的稀疏索引排序：从 LS 取元素→桶化→生成直方图→写回）→ MLU 做 layout 变换 → FI 经 NoC 与 HBM/片上 SRAM 交换数据（字节对齐切片 DMA、索引 DMA）。异步 dataflow 使数据移动与计算按依赖就绪执行、CB 依赖跟踪保证生产者-消费者正确性。冗余行：每 PE 列容忍 1 个坏 PE（boot 时替换，软件透明）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PE 为 RTL 硬核（论文未给面积/工艺分解）；编程走"RISC-V 核 + 固定功能单元自定义指令"的数据流模型（类似 TPU/MTIA 系指令式 + dataflow 混合），kernel 用 C++/Triton 编写、经 MTIA 编译器（TorchDynamo/AOTAutograd/TorchInductor）生成。使用场景：DLRM 训练（GEMM、TBE embedding、Shampoo 优化器、radix-sort 反向）与 LLM 推理（GEMM/SIMD 算子）。评估：TBE 前向 2.0×/1.6×、反向 2.1×/1.6×（vs H100/H200）；BF16-add 5.57 TB/s；BF16-GEMM 大矩阵 59% roofline 效率（H100 63%）。信息缺口：论文未给出 DPE 的 MAC 阵列微架构图与 RISC-V 核的具体型号。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
