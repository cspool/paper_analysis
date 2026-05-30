## SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

- baseline方法是什么？
  - **Baseline 方法**：FlashAttention2 / xformers 使用 FP16 精度在 GPU 上通过 tiling + online softmax 做 exact attention，以及 SageAttention2 使用 INT8 per-block quantization 加速 attention。这些工作在推理中都受限于 FP16/INT8 Tensor Core 的理论吞吐上限（RTX5090 上 FP16 ≈ 200 TOPS，INT8 ≈ 800 TOPS）。
  - **Baseline 全栈执行例子（推理）**：
    - 算法层：FlashAttention2 的 FP16 QK^T → online softmax → FP16 PV，或 SageAttention2 的 INT8 QK^T → online softmax → INT8 PV（per-block 量化 + smoothing K/Q）
    - 系统框架层：论文未明确说明（plug-and-play 替换 attention 实现）
    - 编译框架层：论文未明确说明
    - Kernel 调度层：FlashAttention2 CUDA kernel 使用 warp-specialized tiling，FP16 MMA 指令；SageAttention2 使用 INT8 MMA + per-thread INT4 PV
    - 硬件架构层：RTX5090 Blackwell GPU，FP16 Tensor Core ~200 TOPS，INT8 Tensor Core ~800 TOPS。FP4 Tensor Core 达 ~1600 TOPS 但在 baseline 中未被利用
  - **Baseline 全栈执行例子（训练）**：
    - 算法层：FlashAttention2 的 FP16/BF16 前向 QK^T → softmax → PV，反向 dV = P^T dO, dP = dO V^T, dS = softmax_backward, dQ = dS K, dK = dS^T Q，全 FP16
    - 系统框架层：论文未明确说明
    - 编译框架层：论文未明确说明
    - Kernel 调度层：FlashAttention2 CUDA kernel forward+backward in FP16
    - 硬件架构层：RTX4090，FP16 Tensor Core

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **方法**：SageAttention3 提出两项创新——(1) FP4 microscaling attention 用于推理加速；(2) INT8 可训练 attention (SageBwd) 用于训练加速。
  - **解决 C1（FP4 值域限制）**：使用 NVFP4 microscaling quantization（1×16 块粒度，E2M1+E4M3 scale），相比 per-tensor/per-token 量化有效抑制 outlier 影响。选择 NVFP4 而非 MXFP4，因为 NVFP4 的 1×16 块大小和 E4M3 scale 在 attention 量化中精度更高。
  - **解决 C2（P 的 scale factor 精度损失）**：提出 two-level quantization——先 per-token 归一化 P 到 [0, 448×6]（level 1，在 FP32 中无损），再做 FP4 microscaling 量化（level 2），使 scale factor s_P 充分利用 E4M3 的 127 个有效表示值（vs 直接量化的 35 个），显著降低量化误差。
  - **解决 C3（训练中梯度敏感）**：识别反向 5 个 MatMul 中 dOV^T 的精度最关键（其误差在 FlashAttention 循环中沿序列长度累积到 dQ/dK），保持 dOV^T 在 FP16，其他 4 个 MatMul 量化到 INT8。选择 INT8 而非 FP8，因 INT8 在反向梯度精度更高且硬件支持更广泛。
  - **论文方法全栈执行例子（推理）**：
    - 算法层：FP4 microscaling QK^T（NVFP4, 1×16 块）→ Smoothing Q/K → online softmax → two-level FP4 quantization of P → FP4 microscaling PV
    - 系统框架层：论文未明确说明（plug-and-play 替换 existing attention）
    - 编译框架层：论文未明确说明
    - Kernel 调度层：CUTLASS+CUDA 实现，包含三项硬件优化：K permutation（对齐 FP4MMA accumulator 与 operand 寄存器布局）、Reuse shuffle（softmax 与 P 量化共享 rowmax）、Producer warp epilogue（双 producer warp ping-pong 实现 MatMul 与 global store 的 overlap）
    - 硬件架构层：RTX5090 Blackwell FP4 Tensor Core，NVFP4 FP4MMA 指令 ≈ 1600 TOPS，实测达 1038 TOPS

  - **论文方法全栈执行例子（训练）**：
    - 算法层：前向 INT8 per-block QK^T + per-token P + per-block V；反向保持 dOV^T 在 FP16，其余 dS K、dS^T Q、P^T dO 量化到 INT8 per-block
    - 系统框架层：论文未明确说明
    - 编译框架层：论文未明确说明
    - Kernel 调度层：OpenAI Triton 实现 forward+backward INT8 attention kernel
    - 硬件架构层：RTX4090 INT8 Tensor Core，前向 2× 加速，反向 1.2~1.6× 加速
