## Tensor Core (TC)

术语是什么？

Tensor Core (TC) 是 NVIDIA GPU 中专用于矩阵乘累加（MMA, Matrix Multiply-Accumulate）的硬件单元，自 Volta 架构（V100, 2017）开始引入。TC 执行单个指令即可完成 D = A × B + C 的分块矩阵运算。TC 的 shape 由 mma_m × mma_n × mma_k 三个维度定义（分别表示输出矩阵 C 的行数、列数、以及 A×B 的 reduction 维度）。现代 GPU 上 TC shape 的演变：Volta 仅支持 m8n8k4（FP16）；Ampere (A100) 支持 m16n8k16 (FP16)、m16n8k8 (TF32)、m8n8k4 (FP64)；Hopper (H100) 在此基础上增加 m16n8k16 (FP64)、以及 WGMMA 模式 m64nNk16 (N∈[8,256])。不同 shape 的吞吐量和延迟差异显著——H100 上 m16n8k16 达 494 TFlops/24 cycles，而 m16n8k8 仅 368 TFlops/16 cycles。TC computation 中，输入数据在参与线程间均匀分布，每个线程持有矩阵分块片段，利用 warp 内 shuffle 指令做归约。Volta 时代的 m8n8k4 在 A100/H100 上不再由真 TC 硬件执行，编译器 fallback 到 ALU 指令（CUDA Cores），这是 DASP 选型 m8n8k4 在现代 GPU 上性能差的根因。

从硬件架构角度拆解术语：

TC 的硬件执行流程（以 m16n8k16 FP16 在 A100 上为例）：
1. **数据加载**：每个 warp（32 threads）协作加载 A（16×16 子矩阵 FP16）和 B（16×8 子矩阵 FP16）到寄存器文件
2. **MMA 执行**：TC 硬件在一个指令周期内完成 16×16×16 = 4096 次 FP16 乘法和 FP32 累加，输出 16×8 FP32 结果矩阵
3. **输出写回**：累加结果经 warp 内线程分布的寄存器写回
4. **与 CUDA Core 对比**：TC 通过单指令完成多线程协作计算，数据重用更好（每个 A 元素被 8 个 B 列复用、每个 B 元素被 16 个 A 行复用），per-thread memory load 低于 CUDA Cores，对 memory-bound kernel（如 SpMV）尤为有利

术语一般如何实现？如何使用？

TC 通过 CUDA PTX MMA 指令（如 `mma.sync.aligned.m16n8k16`）或更高级的 WMMA API（`nvcuda::wmma`）编程。CUTLASS 模板库提供更高层次的 TC kernel 抽象。通常将矩阵分块为与 TC shape 对齐的 tile，通过 shared memory 做数据 staging 提升数据重用，使用 async-copy（A100+）或 TMA（H100）异步传输 tile 数据以 overlap 计算和 memory transfer。性能 profiling 使用 NVIDIA Nsight Compute 检查 TC 利用率。

在量化推理中，TC 的 INT4 和 INT8 模式至关重要——Ada Lovelace (RTX 4090) 的 INT4 TC 峰值吞吐为 FP16 的 8×，INT8 为 FP16 的 2×。RoMeo 利用 INT4 TC 完成主体计算（~88%），用 INT8 TC 处理 outlier token 的高精度计算（~12%），geomean kernel speedup 达 4.68× over BF16。在 Drawloom 中，ArbitWeave 策略利用 m16n8k16 替代 m8n8k4，避免 ALU fallback 并利用现代 GPU 的高效 TC shape。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

