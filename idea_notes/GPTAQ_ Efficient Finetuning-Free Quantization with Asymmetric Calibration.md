## GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration

- baseline方法是什么？
  Baseline 是 **GPTQ**（Frantar et al., 2022），基于 Optimal Brain Compression (OBC) 框架的经典 PTQ 方法。GPTQ 每层独立执行**对称校准（Symmetric Calibration）**：最小化 `||(w+Δw)X − wX||²`，其中 X 来自前一层的量化输出。由于前层量化误差已改变激活分布，X 与全精度模型的输入 X̃ 存在偏差（即 ΔX = X̃ − X ≠ 0），且该偏差沿网络深度累积（Fig. 2a 显示激活 MAE loss 逐 block 持续增长）。GPTQ 忽略了这一偏差，仅优化当前层的局部 MSE。

  GPTQ 全栈执行例子（LLaMA2-7B, W4A4, 单卡 A100）：
  - **算法pipeline**：校准数据 X（128 sequences × 2048 tokens from WikiText2）→ 逐层：计算 Hessian H = XX^T → Cholesky 分解 H^{-1}=LL^T → 按 block size B 逐列量化：Q_{:,j} ← quant(W_{:,j}) → 误差补偿 δW = −(W_{:,j}−Q_{:,j})/L_{jj} · L_{j,j:}^T（仅二阶项）→ lazy-batch 更新后续列。激活量化在权重量化后应用（QuaRot 风格），目标为 `||ŵX − wX||²`，不对齐全精度输出。
  - **系统框架**：HuggingFace Transformers + PyTorch，自实现 GPTQ 脚本，无 Serving 框架修改。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel（matmul、cholesky），无自定义 kernel。推理使用标准 INT4 dequantize + FP16 GEMM。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **对称校准忽略前层误差累积**：GPTQ 每层假设输入激活 X 已经正确（即等价于全精度前向的 X̃），但前层量化后的实际 X 与 X̃ 存在偏差 ΔX，该偏差沿网络深度累积，导致深层量化严重偏离全精度模型行为。
  2. **校准目标是"局部最优"而非"全局最优"**：最小化 `||ŵX − wX||²` 确保当前层输出与传统前向一致，但传统前向的输入 X 已经是"错的"（受前层量化误差影响），导致最终模型输出与全精度模型偏差大。
  3. **在低比特场景下偏差急剧放大**：W2A4 时 GPTQ 对 LLaMA 模型仍退化严重，RTN 直接崩溃（PPL > 1000）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **GPTAQ**，将校准目标从对称的 `||ŵX − wX||²` 改为非对称的 `||ŵX − wX̃||²`（X̃ 为全精度模型的输入激活），并推导出高效的闭式解实现：

  **(1) 非对称校准框架（解决缺陷 1, 2）**
  对称校准时优化目标中的 target 是 wX，即"量化层在已有误差输入下的输出"。非对称校准时 target 改为 wX̃，即"全精度层在全精度输入下的输出"。引入残差 r = wX̃ − wX = wΔX（输入激活偏差在输出空间的投影），通过 Lagrangian 求导得最优权重更新：
  ```
  Δw = −(ŵ_q − w_q)/H_{qq}^{-1} · H_{q,:}^{-1} + r X^T H_{-q}^{-1}
  ```
  第一项是 GPTQ 的量化误差补偿，第二项是 GPTAQ 新增的**残差误差补偿项**，显式将前层累积误差通过 Hessian 逆回传到当前层权重。消融实验（Table 5）验证：仅用第一项 = GPTQ（WikiText2 7.80），仅用第二项名曰 GPTAQ'（WikiText2 7.97），两项联合 = GPTAQ（WikiText2 7.36，最优）。

  **(2) 残差分解避免重复计算（解决效率瓶颈）**
  直接计算残差 r 需每次迭代重新评估 R = W X̃ − W X，复杂度 O(mnk) 极高（k ≈ 128×2048 >> n）。GPTAQ 利用 R = Σ_{q=1}^n W_{:,q} ΔX_{q,:} 将残差分解为 n 个独立神经元分量。预计算一次 ΔX 后，第 q 次迭代仅关注第 q 个神经元残差分量 `W_{:,q} ΔX_{q,:} X_{:,q:}^T H_{-q}^{-1}`，复杂度从 O(mnk) 降至 O(mn)，无需重复计算 R。

  **(3) Cholesky 重构化 + 矩阵融合实现 GPU 并行（解决数值稳定性和效率）**
  通过 **Lemma 4.1**：Cholesky 因子 L 的子矩阵 L_{q+1:,q+1:} 即为消去前 q 行的逆 Hessian H_{-q}^{-1}，替代数值不稳定的逐次 Gaussian Elimination。通过 **Theorem 4.2** 将 P 矩阵（存储每行残差补偿项）计算融合为一行 GPU 友好代码：
  ```
  P = ((ΔX X^T L) ⊙ M_U) L^T
  ```
  其中 M_U 是严格上三角掩码矩阵。利用 CUDA 高度并行，计算 P 仅需 <1ms（vs 非并行实现 >10⁴× slower，Fig. 4a）。

  **(4) Lazy-Batch 更新（解决 GPU 利用率）**
  与 GPTQ 的 lazy-batch 策略一致：block 内逐列更新，block 后批量更新 block 外列。GPTAQ 的 block 外更新为：
  ```
  ΔW_{:,Q:} = E · L_{Q,Q:}^T + W_{:,Q} · P_{Q,Q:}
  ```
  两项融合在同一 kernel 中。GPTAQ 整体额外延迟：n<4096 时 <10%，n>4096 时 30-40%（Fig. 4b）。

  论文方法全栈执行例子（LLaMA2-7B, W4A4, 单卡 A100）：
  - **算法pipeline**：校准数据 128 sequences → 先全精度前向收集每层 X̃（FP 输入），同时记录量化后输入 X → 计算 ΔX X^T 和 H → 启用激活量化（A→W 顺序，Table 6 证明此顺序对 GPTAQ 更优）→ 逐 block：对每层 ① Cholesky 分解 L ← H + λI → ② P ← ((ΔX X^T L) ⊙ M_U) L^T → ③ 按 block 逐列量化 + GPTQ 项补偿 + GPTAQ 残留项补偿。整模型流程（Algorithm 2）：每次仅一个 transformer block 在 GPU，X̃ 临时存储可释放，内存瓶颈为 P 矩阵（n×n FP16），LLaMA2-7B 每层 P 约占 0.16-0.70GB。
  - **系统框架**：HuggingFace Transformers + PyTorch，基于 GPTQ 代码修改（仅多约 20 行）。单卡 A100 量化 LLaMA3.1-405B（126 blocks, intermediate=8096）。W4A4 LLaMA3-8B 耗时 0.3 GPU-hours（GPTQ 0.2h），LLaMA3-70B 耗时 2.7 GPU-hours（GPTQ 1.8h）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：标准 PyTorch CUDA kernel（Cholesky、matmul、element-wise mask）。无自定义 Triton/CUDA kernel。推理使用标准 INT4 dequantize + FP16 GEMM。
  - **硬件架构**：NVIDIA A100 GPU，无自定义硬件。

  关键设计动机映射：
  - GPTQ 对称校准假设 X = X̃，忽略前层误差累积 → GPTAQ 非对称校准目标 `min ||ŵX − wX̃||²`，显式补偿 r X^T H_{-q}^{-1}
  - 直接非对称优化需每次迭代重算 R（O(mnk)）→ 残差分解 R = Σ W_{:,q} ΔX_{q,:}，复杂度降至 O(mn)
  - Gaussian Elimination 数值不稳定 → Cholesky 重构化（L_{q+1:,q+1:} 等价于 H_{-q}^{-1}）+ Theorem 4.2 矩阵融合
  - W2A4 下 GPTQ/RTN 严重退化 → GPTAQ 在 W2A4 LLaMA 上将 GPTQ PPL 降低 20%-90%
  - 实现仅比 GPTQ 多 ~20 行代码，易部署和复现
