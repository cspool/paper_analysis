## VAR-Turbo Accelerator

术语是什么？
VAR-Turbo Accelerator 是论文提出的面向 Visual Autoregressive 图像生成的专用 ASIC 加速器，在 TSMC 28nm+HPC 工艺下实现，面积 7.09 mm²，功耗 1.98 W，片外使用 2×64bit HBM2 channel @2GHz（32GB/s）。加速器由 Unified Attention Core、Radix Sort Core、MLP Core、Non-Linear Core、SIMD Core、global/weight memory、DMA、AXI interconnect 和控制器组成，协同执行 PD（Draft-Free Parallel Decoding）、TA（Token Aggregation）和 DB（Dynamic Bypass）三项算法优化的完整 VAR-Turbo 推理 pipeline。

从硬件架构角度拆解术语：
VAR-Turbo Accelerator 的硬件组成和运转流程（以一次 PD iteration 为例）：
1. DMA 从 HBM2 加载 token sequence 到 global memory
2. MLP Core 执行 QKV projection、FFN 等矩阵运算
3. Unified Attention Core 根据当前层所在 Region 切换 dataflow：
   - Learning Region（浅层 SA mode）：Snooper 配置 PE Cell 为 OP dataflow→Small Attention 聚合 local window→再切换 BA mode（Row dataflow）→Big Attention 全局建模
   - Inert Region（深层 BA mode）：Row dataflow 执行标准 attention
4. Non-Linear Core 执行 softmax、GELU 等非线性操作
5. SIMD Core 处理 element-wise 运算
6. Radix Sort Core 在 PD 阶段执行 confidence TopK（CountBin→PrefixSum→SelectBin→Filter），在 DB 阶段执行 importance score TopK
7. 层间 producer-consumer schedule 通过 DMA 和 AXI interconnect 协调各 core 的流水线执行
与 V100 GPU 对比（scale-up 建模 14 TFLOPS、512GB/s）：平均 210.3× speedup、423.5× energy efficiency improvement。

术语一般如何实现？如何使用？
RTL 以 SystemVerilog 实现，Synopsys Design Compiler 在 TSMC 28nm 综合，DCG+ICC2 完成 placement/routing 生成 layout 和 netlist，经 DRC/LVS 后用 VCS+PrimeTime 做 post-layout simulation 获取功耗和延迟。与 baseline ViTCoD/AdapTiV（ViT accelerator）在相同工艺下重新综合以公平对比。Attention Core 和 Memory 为主要面积/功耗来源；Radix Sort Core 仅占 4.9% 面积和 6.3% 功耗。

涉及论文标题：
- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

