## 混合精度可重构 PE（Mixed-precision Reconfigurable PE，INT8/FP16 由 INT8 分解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVA 的基座 GEMM 单元采用 INT8 PE（现代加速器标准），但 LLM decode 需要 FP16 保精度（weight-only 量化下激活/计算 FP16），prefill 只需 INT8。为避免单独 FP16 乘法阵列的硬件开销，EVA 复用 INT8 MAC 阵列经轻量重配支持 FP16：FP16 = 1 符号位 + 5 指数位 + 10 尾数位，把一次 FP16 乘法分解为 4 个 INT8 乘法重构尾数乘法，加 6-bit 加法器与若干 XOR 处理指数与符号；FP16 加法前插对齐单元（alignment）并复用 INT32 累加器。32×32 INT8 阵列可动态重配为 32×8 FP16 阵列，正好匹配 VQ-GEMM 的 tiling（v=32, d=8），实现 prefill（INT8）与 decode（FP16）无缝切换。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
FP16 乘法分解（尾数 10-bit×10-bit → 拆为 4 个 INT8 乘法分段重构 + 移位求和）+ 6-bit 加法器（指数偏移）+ XOR（符号）；FP16 加法需先指数对齐（alignment 单元）再入 INT32 累加器。配置切换：decode 用 32×8 FP16（input-stationary），prefill/attention 用 32×32 INT8（weight-stationary），同一阵列硬件复用、无缝切换。每周期一个 PE 做 1 次 FP16 或 4 次 INT8 运算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：RTL（Verilog HDL）实现，Cadence Genus + TSMC 28nm @500MHz 综合（EVA 面积 1.414mm²、功耗 3.117W，比纯 INT8 阵列的增量主要来自 FP16 支持）；SRAM buffer 用 Cacti 7.0（28nm）建模，DRAM 功耗用 DRAMsim3。类似思路：多精度 PU（尾数乘法器分解）、NVIDIA Tensor Core 的 FP16/INT8 支持。使用方式：同时服务 decode（VQ-GEMM FP16）与 prefill（INT8 GEMM）的双阶段 LLM 加速器，避免双阵列面积开销。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
