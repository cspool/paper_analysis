## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ResQ 使用 CUDA 11.8 + CUTLASS 实现混合精度推理 kernel，利用 TensorCore 执行 INT4 和 INT8 GEMM 操作。kernel 实现的分块矩阵乘法中，低精度（4-bit）和高精度（8-bit）操作数分别走各自精度的 GEMM kernel，通过 ResQ 的投影融合（U_A 融入权重）和运行时投影（U_C 8-bit 量化，U_D Hadamard 快速变换）保证正确性。实验比较 ResQ kernel 与 16-bit baseline 以及纯 INT4 kernel 在 NVIDIA RTX 3090 上单 decoder block 的性能加速比。

- 后端平台：NVIDIA RTX 3090 GPU（resource-constrained 场景代表）。CUDA 11.8 + PyTorch。

- 评估性能的软件/脚本：使用 CUTLASS (https://github.com/NVIDIA/cutlass) 实现 INT4/INT8 GEMM 在 TensorCore 上的运算。测试单 decoder block 在不同模型和序列长度下的延迟，对比 16-bit FP baseline 与纯 INT4 kernel。论文未提供具体 benchmarking 脚本。

- 开源情况：代码开源 https://github.com/utkarsh-dmx/project-resq。CUDA kernel 实现在开源仓库中，包含 CUTLASS 集成的 INT4/INT8 GEMM 调用。

- 评估原理和 kernel 输入到性能输出的全过程：
  1. **输入**：量化后的模型权重（W_q = Q_L(U_l^T·W) + Q_H(U_h^T·W)，已离线完成）和运行时激活 X。
  2. **前处理**：激活 X 通过已融入前一层权重的 U_A 自动投影；若为注意力块内，key/query 先经 U_C 显式投影（8-bit 量化）；若为 FFN 块内，经 U_D Hadamard 变换。
  3. **GEMM 执行**：调用 CUTLASS INT4 GEMM kernel 计算 Q_L(XU_l)·Q_L(U_l^T·W)；调用 CUTLASS INT8 GEMM kernel 计算 Q_H(XU_h)·Q_H(U_h^T·W)。两路结果在 INT32 累加器中求和得到输出。
  4. **性能测量**：在 RTX 3090 上运行，batch size=1，测量 decoder block 的总延迟（含 GEMM 和投影开销）。加速比：1.61× 到 3.03× 相比 16-bit baseline（含 Hadamard 变换开销）。相比纯 INT4 kernel 仅慢 14%。
  5. **结果**：更大模型和更短序列获得更高加速比。混合精度计算和运行时投影的额外开销较小。
