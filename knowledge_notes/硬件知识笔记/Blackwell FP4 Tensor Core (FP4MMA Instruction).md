## Blackwell FP4 Tensor Core (FP4MMA Instruction)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Blackwell FP4 Tensor Core 是 NVIDIA Blackwell 架构 (Compute Capability 10.0/SM100, B200/B300/RTX5090) 原生支持的 4-bit 浮点矩阵乘累加硬件单元。每条 FP4MMA 指令接受两个 packed FP4 矩阵（使用 NVFP4 或 MXFP4 格式）及其对应的 FP8 block-scale factor，在 Tensor Core 内部自动完成 dequantization + 矩阵乘累加，输出 FP32 accumulator 结果。该指令无需软件 dequantization，将 4-bit 数据的硬件利用率提升至接近理论峰值。Blackwell B200 的 FP4 Tensor Core 理论吞吐为 20 PFLOPS（FP16 的 4×），RTX5090 上 FP4 microscaling MatMul 约 1600 TOPS（FP16 约 200 TOPS，8× speedup）。SageAttention3 在 RTX5090 上实测达到 1038 TOPS。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FP4MMA 在 attention kernel 中的执行流程（SageAttention3 on RTX5090）：
1. **数据准备**：Q, K, V 以 FP16 加载到 SM 的 shared memory。Quantization kernel 将 Q, K, V 按 NVFP4 格式（E2M1, 1×16 block）pack 为 INT32（每 8 个 FP4 元素 1 个 INT32），scale factor 以 E4M3 FP8 格式存储在独立数组中。
2. **FP4MMA QK^T**：Consumer warp 将 packed Q̂ (FP4 NVFP4), s_Q (FP8), K̂ (FP4), s_K (FP8) 加载到 Tensor Core fragment registers → FP4MMA 指令执行 → Tensor Core 内部：每个 1×16 block 的 scale factor 广播到 16 个乘法器，恢复 FP4 元素到等效 FP16 值域 → 乘累加 → FP32 accumulator 写入寄存器。
3. **Layout 转换**：FP4MMA 的 FP32 accumulator 内存布局与 operand A (Q̂) 的寄存器布局不匹配（Fig 19 vs Fig 20）。SageAttention3 通过 permute accumulator layout（Fig 21）+ 对应重排 K 的列（fuse 到量化 kernel）解决，避免 thread shuffle。
4. **FP4MMA PV**：同理，P 经过 two-level quantization 量化为 NVFP4 → P̂ (FP4) + s_P2 (FP8) 与 V̂ (FP4) + s_V (FP8) 送入 FP4MMA → FP32 accumulator 输出 → post-scale by s_P1 (FP32) → O output。
5. **对比 Ampere/Hopper**：Ampere (SM80) 仅支持 INT8/FP16 Tensor Core，Hopper (SM90) 新增 FP8 Tensor Core。Blackwell 的 FP4 Tensor Core 将 4-bit 计算的硬件支持从软件模拟推进到原生指令级别。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：通过 CUDA PTX 指令直接调用（如 `mma.sync.aligned.m16n8k32.row.col.f32.f4.f4.f32` 或类似 NVFP4 专用指令），或通过 CUTLASS 3.x 的高级抽象层使用。关键约束：(1) FP4 数据必须按 block-size（NVFP4: 1×16, MXFP4: 1×32）pack 并对齐；(2) scale factor 的 layout 需与 mma 指令的参数期望匹配；(3) accumulator 和 operand 的寄存器 fragment layout 不同，需要 permutation 或 thread shuffle 对齐。使用场景：所有需要极低比特推理的 Transformer attention/LayerNorm/GEMM，Blackwell 架构 GPU 独占。NVIDIA 官方文档：https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
