## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 D2MoE 的 **Parallel Loading Dequantization Kernel**，针对 MWQ 嵌套量化结构在边缘 GPU 上的反量化性能优化。核心包括：(1) **加载并行性**：将量化数据从磁盘直接传输到 GPU global memory，与激活值从 global memory 向 L2 cache 的移动重叠（DMA engine + CUDA stream 并行）。(2) **计算并行性**：expert 反量化在 CUDA Cores 上执行，与 Tensor Cores 上的 expert 矩阵计算同步进行（Figure 8），利用 GPU 分离的 CUDA Core 和 Tensor Core 资源。(3) **优化 binary operation**：借鉴 Any-Precision LLM [29]，避免传统 bit-transpose 方法中 INT → FP 的多轮类型转换链，binary residual (±1) 通过单个 bit extract + conditional sign assignment 实现，每 element 仅需 1 次 FMA。

  实验比较（Figure 12，dequantization overhead 分析）：D2MoE-V1 在 LLaMA-MoE-3.5B 和 Mixtral 8×7B 上测量 dequantization 的计算开销、峰值内存开销和延迟开销。4 requests 时计算开销 20.77%、延迟开销 18.56%；32 requests 时因 MWQ 嵌套权重复用增加分别降至 16.77% 和 5.3%。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 3060 (6GB, Ampere, 3584 CUDA Cores, 112 Tensor Cores) 和 Jetson AGX Orin 64GB (2048 CUDA Cores, 64 Tensor Cores)。CUDA 基于 Ampere 和 Ada Lovelace 架构。GPU 存储层级：NVMe SSD (3.5 GB/s) → GPU Global Memory → L2 Cache → Shared Memory → Registers。使用 Triton 进行 I/O-compute 并行编程。

- 评估性能的软件/脚本是什么。修改了什么。
  自研 D2MoE 引擎 (~2,500 LOC Python + CUDA)。主要修改：(a) 实现 MWQ 专用 CUDA dequantization kernel，将 per-group scale/zero-point 应用与 binary residual 累加融合为单一 kernel；(b) 使用 CUDA stream 异步重叠 disk→GPU 数据传输与 dequantization kernel 执行；(c) 利用 Triton 协调 dequantization + expert FFN GEMM 的 pipeline 执行。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供公开开源代码仓库。Kernel 原理如下：

  ```
  === MWQ Dequantization Kernel 全流程 ===

  Input (per expert, loaded from disk to GPU global memory):
    Q_W_b1: int8 [s, h] (b_1-bit quantized, e.g. INT2 packed)
    z_b1: int8 [s, h/128] (per-group zero points, group_size=128)
    s_b1: fp16 [s, h/128] (per-group scales)
    Q_W_bk: packed 1-bit [s, h] (binary residual for k=2..K)
    s_bk: fp16 [s, h/128] (per-group scales for k=2..K)

  Step 1: Parallel Loading (CUDA Stream I/O)
    cudaMemcpyAsync(Q_W_b1..bK, disk, sizes, H2D, load_stream)
    activation X moves: global mem → L2 cache (comp_stream)

  Step 2: Dequantization Kernel (CUDA Cores, per-group parallel)
    for each group g_id (128 elements):
      # Base asymmetric dequant to FP16
      for idx in group:
        W_fp16[idx] = (int(Q_W_b1[idx]) - int(z_b1[g_id])) * s_b1[g_id]
      
      # Binary residual accumulation (k = 2..K):
      for k in 2..K:
        for idx in group:
          sign_bit = (Q_W_bk_packed[idx/8] >> (idx%8)) & 0x01
          W_fp16[idx] += (sign_bit ? 1.0 : -1.0) * s_bk[g_id]
          # 仅 1 次 bit extract + 1 次 FMA per element per bit-level
          # vs 传统 bit-transpose: unpack → int8→int32→fp32→fp16 (4 ops)
      
      store W_fp16[group] to shared memory

  Step 3: Expert FFN (Tensor Cores, overlaps with Step 2 of next expert)
    for expert e in I/O-complete queue:
      GEMM(W_fp16 @ X) using Tensor Core FP16 MMA
      # CUDA Cores dequantize next expert while Tensor Cores compute current

  === 传统 bit-transpose vs 本文 binary ops ===
  传统 (per INT2):
    unpack_2bit(packed) → int8_val → int32_val → fp32_val → fp16_val → dequant
    (5+ operations per element, multiple type conversions)
  本文 (binary residual path, k≥2):
    bit_extract(packed) → sign → fp16_val += sign * scale
    (2 operations: 1 logical + 1 FMA, zero type conversion)
  ```

  性能：dequantization overhead 随 request 数增加从 ~20% 降至 ~5%（权重复用），临时 FP16 内存立即释放，不影响 peak memory。
