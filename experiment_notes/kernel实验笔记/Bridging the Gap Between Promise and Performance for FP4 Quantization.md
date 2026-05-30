## Bridging the Gap Between Promise and Performance for FP4 Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现 QuTLASS v1.0，一套面向 NVIDIA Blackwell GPU 的高性能低精度量化 kernel 库，基于 NVIDIA CUTLASS 构建。包含两类 kernel：
    1. **Quantization-related kernels**：轻量级 fused kernel，实现在线 block-wise Hadamard 旋转 + 量化 + scale 计算的融合。支持 k∈{16,32,64,128} 的 block diagonal 矩阵。对 k<256，dense 变换仍为 memory-bound，任意旋转矩阵（非仅 Hadamard）可在运行时加载实现同成本运行。量化方法支持 MSE 和 Abs-Max，模板设计便于扩展。
    2. **Matmul-related narrow precision kernels**：处理 FP4 量化与矩阵乘法间的硬件强制 scale 重排（tcgen05.mma 要求），通过 Triton kernel 实现。Matmul 支持多后端（CUTLASS, FlashInfer），灵活插拔。
  - 实验比较：在 B200 和 RTX 5090 上测量单层 throughput（TFLOPS），对比 "ideal"（纯 FP4 matmul 上限）和 "actual"（含 Hadamard/量化/scale 计算开销）。端到端速度在 vLLM 中测量，MXFP4 vs BF16 baseline，不同 batch size（1-256）。

- 后端平台是什么，配置是什么。
  - NVIDIA B200 GPU（Blackwell SM100 架构）。
  - NVIDIA RTX 5090 GPU（Blackwell SM120 架构）。
  - CUDA/CUTLASS 框架 + FlashInfer 后端 + Triton kernel（scale 重排）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：vLLM 框架集成 QuTLASS kernel，测量 Llama-3.3-70B 端到端推理吞吐。
  - 修改内容（关键 kernel 设计）：
    1. **Fused Online Rotation Kernel**：将 MR-GPTQ 的激活端 Hadamard 旋转与量化+scale 计算融合为单个 kernel。Hadamard 对 k<256 的 block 为 memory-bound，因此旋转矩阵可运行时加载。Epilogue 直接完成量化，避免中间 DRAM 写入。
    2. **Scale Rearrangement Kernel**：Blackwell tcgen05.mma 要求特定的 scale factor layout，QuTLASS 用 Triton kernel 在 FP4 量化后、矩阵乘前完成硬件强制的 scale 重排。
    3. **Multi-backend Matmul**：支持 CUTLASS 和 FlashInfer 后端，根据 workload 和硬件灵活选择。B200 上 MXFP4 的 matmul throughput *超过* NVFP4（~15%），得益于 power-of-two scales 和更大 group size 降低 overhead。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 代码开源：https://github.com/IST-DASLab/qutlass
  - Kernel 执行全过程（以 Llama-3.3-70B MXFP4 单层推理在 B200 为例）：

  **阶段 1: 预处理（离线）**
  - 权重 W 经 MR-GPTQ 量化完成：W_q = MXFP4_quant(W · H_k)，其中 H_k 为 block_size=k 的 Hadamard 旋转（k=32 for MXFP4）。旋转已融合入量化权重，权重以 MXFP4 packed 格式存储。

  **阶段 2: 推理时 Fused Rotation + Quantization（在线）**
  - 输入：FP16 激活 X ∈ R^{M×K}，block Hadamard 矩阵 H_k
  - 步骤 2a: 加载 H_k 到寄存器（k<256，Dense 矩阵，memory-bound，可在运行时任意加载）
  - 步骤 2b: X_rot = X @ H_k（block-wise 旋转，每 k×k block 独立）
  - 步骤 2c: 计算 per-group scale s_G = absmax(X_rot per G=32 elements)（fused epilogue）
  - 步骤 2d: X_q = FP4_quantize(X_rot / s_G)（fused epilogue，直接输出 E2M1 4-bit 值 + E8M0 scale）
  - 输出：MXFP4 量化激活 X_q + per-group scales

  **阶段 3: Scale Rearrangement（硬件强制）**
  - 输入：per-group scales（原始 group layout）
  - 步骤：Triton kernel 将 scales 重排为 tcgen05.mma 要求的 layout（block scaling factors layout，参照 cuBLAS 文档）
  - 输出：重排后的 scales

  **阶段 4: FP4 Matrix Multiplication（硬件加速）**
  - 输入：MXFP4 packed 权重 W_q、MXFP4 量化激活 X_q、重排后的 scales
  - 步骤：Blackwell tcgen05.mma 指令执行 FP4 矩阵乘法
  - 输出：FP16/BF16 格式的输出 activation

  **阶段 5: 性能输出**
  - "Ideal" 曲线：仅测量 tcgen05.mma matmul throughput（不含步骤 2-3 开销）
  - "Actual" 曲线：包含步骤 2-4 全部开销
  - B200 单层 speedup（vs FP16）：MXFP4 ≈ 3.6×（ideal 4×），NVFP4 ≈ 3.0×
  - RTX 5090 单层 speedup：MXFP4 ≈ 6×（ideal 8×）
  - MXFP4 B200 上比 NVFP4 高 ~15% throughput（power-of-two scales + 更大 group size 降低 overhead）
  - B200 端到端 vLLM Llama-3.3-70B speedup：MXFP4 vs BF16 = up to 2.2×（batch size=1-256）
  - RTX 5090 端到端 speedup：nearly 4×
