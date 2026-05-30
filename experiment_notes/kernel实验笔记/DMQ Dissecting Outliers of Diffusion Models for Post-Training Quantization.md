## DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  定制 CUDA kernel 实现了 W4A8 GEMM，集成量化（quantization）、权重 bit-shifting、GEMM 和反量化（dequantization）的融合操作。Section E 中展示了该自定义 kernel 与 PyTorch FP32 GEMM 的延迟对比（Figure 8），在 M=3072 时达到 5.17× 加速。bit-shifting 引入的开销极小（仅在权重加载后执行 Ŵ^{shifted}_{kj} = Ŵ_{kj} ≪ δ_k），且 PTS 仅应用于 skip connection 层（网络总层数的小子集），对整体延迟影响微乎其微。

- 后端平台是什么，配置是什么。
  论文未明确说明 GPU 型号。实验基于 PyTorch + 自定义 CUDA kernel。PTS 的 bit-shift 操作在 GPU kernel 执行时于权重加载后立即完成，验证了 2 的幂次缩放在硬件上的高效性。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：自定义 CUDA kernel 与 PyTorch FP32 GEMM 直接对比延迟。修改内容：kernel 将量化（MinMax Q）→ bit-shift on weight → INT GEMM → dequantization 融合为单次 kernel launch，消除中间结果的 DRAM 写入。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未说明 kernel 代码是否开源（https://github.com/LeeDongYeun/dmq 主要包含量化算法代码）。kernel 执行流程：

  **输入**：INT4 packed 权重 W̃ ∈ Z^{Cin×Cout}，INT8 激活 X̃ ∈ Z^{B×Cin}，scale s^X（标量）、s^W ∈ R^{Cout}，PTS 因子 δ ∈ N^{Cin}
  
  **Kernel 执行全过程**（W4A8 GEMM with PTS）：
  1. 从 global memory 加载 W̃ 行块到 shared memory
  2. **Bit-shift（PTS）**：对每个通道 k，W̃^{shifted}_{kj} = W̃_{kj} ≪ δ_k（左移操作，等价于乘 2^{δ_k}），在寄存器中完成
  3. 从 global memory 加载 X̃ 块到 shared memory
  4. INT8 × INT32 矩阵乘累加：C_ij = Σ_k X̃_ik · W̃^{shifted}_{kj}
  5. **Dequantization**：Y_ij = s^X · s^W_j · C_ij（转 FP32）
  6. 输出 Y 写回 global memory

  **评估原理**：固定矩阵维度（K=N=4096），变化 M 测量延迟。对比 PyTorch FP32 GEMM baseline，W4A8 量化 kernel 因数据位宽减小（4-bit 权重 + 8-bit 激活 vs 32-bit）实现吞吐提升，bit-shift 开销因仅在权重加载阶段执行而极低。
