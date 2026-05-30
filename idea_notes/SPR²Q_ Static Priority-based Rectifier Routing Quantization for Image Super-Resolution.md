## SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution

- baseline方法是什么？
  - Baseline 方法：(a) **传统 PTQ 方法**（PTQ4VM, Quamba, MambaQuant）：均为 quantizer-only 方法——仅优化量化器裁剪界 (a, b) 以最小化 ||x - x_q||，不修改预训练模型权重。量化过程为：固定权重 W → clip(W, a, b) → round → dequant → W_q，模型自身不具备主动适应量化的能力。(b) **CNN/Transformer PTQ 方法**（DBDC+Pac, 2DQuant）：为 CNN 或 Transformer 设计的 PTQ，移植到 Mamba 架构时无法处理 Mamba 的 recurrent state 和 dynamic gating 带来的 error accumulation 和 numerical sensitivity 问题，导致在 SR 任务上细节模糊和纹理丢失。(c) Baseline 的核心缺陷：(1) **仅优化量化器参数，不修改模型权重**——权重本身不包含针对量化的补偿信息，在极端低比特（2-bit）下信息损失严重；(2) **补偿信息单一**——单组补偿参数无法覆盖不同层的异构量化误差模式，产生 homogenization 问题；(3) **Mamba 架构适配缺失**——现有 Mamba 量化方法（Quamba, MambaQuant）主要验证在 classification/language modeling，移植到 SR 时因 pixel-level precision 要求无法满足 fidelity 需求。
  - Baseline 在模型推理全栈的执行例子（以 MambaIRv2-light, PTQ4VM W4A4, ×2 SR 为例）：
    - **算法 pipeline**：预训练 MambaIRv2-light FP32 权重 (3.01MB) → PTQ4VM 仅校准量化器 (a, b)（无权重修改）→ Q_{a,b}(W) = clip(W, a, b) → round → dequant → 得到 W4A4 量化权重 → 前向推理 X @ W_q → 输出 SR 图像。Set5 PSNR=37.17 (vs FP32 38.26, 下降 1.09dB)。
    - **系统框架**：论文未明确说明。
    - **编译框架**：论文未明确说明。
    - **kernel 调度**：标准低精度 GEMM（PaddlePaddle 框架内置），无额外 kernel fusion 或算子优化，量化前后无缓存或调度逻辑。RTX 4090, 无自定义 CUDA kernel。
    - **硬件架构**：NVIDIA RTX 4090 GPU。推理时 MambaIRv2-light 以 75.6G FLOPs 执行，PTQ4VM 仅降低精度不改变计算图结构，FLOPs 与量化后模型参数位宽成比例（4-bit: 22.0G，2-bit: 18.2G）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：SPR²Q = PQFR (Pre-Quantization Fine-tuning with Fused Rectifier) + SPR² (Static Priority-Based Rectifier Routing)。核心思想是"在量化前主动向模型注入补偿信息"，使模型自适应量化过程。(1) **PQFR 解决"仅优化量化器"缺陷**：引入可训练低秩 rectifier ΔW = BA (A∈ℝ^{r×d_in}, B∈ℝ^{d_out×r})，在量化前将 rectifier 增量融合到冻结权重 W' = W + ΔW，再对 W' 进行量化。同时联合优化 rectifier 参数 (A,B) 和量化器参数 (a,b)，使用混合损失（L_pixel 像素级 + L_feature 逐块特征对齐），通过 STE 反向传播。这使权重本身包含了补偿量化误差的先验信息，P只用 PQFR 即从 37.20 提升至 37.44 dB PSNR。(2) **SPR² 解决"补偿信息单一"缺陷**：将单 rectifier 扩展为 N=4 个 rectifier 组成的 rectifier group。训练阶段使用轻量门控网络 g_i 为每个输入动态加权聚合不同 rectifier 的增量（ΔW_fused = Σ g_i · ΔW_i），鼓励各 rectifier 专业化处理不同类型量化误差。离线校准阶段通过梯度下降学习每个模块的最优静态门控权重 ĝ（Eq. 12），构建 SPR²Q Table 记录每个模块的最优增量。推理时每个模块直接从 Table 检索最优 ΔW，离线融合并量化，推理时无任何额外计算。(3) **跨架构泛化**：在 SwinIR-light (Transformer) 上 2-bit ×2 SR 超越 2DQuant (+1.14 dB)、FIMA-Q (+1.22 dB)、APHQ-ViT (+1.14 dB)，验证方法不依赖特定架构。
  - 全栈执行例子（SPR²Q W4A4, MambaIRv2-light, ×2 SR, RTX 4090）：
    - **算法 pipeline**：
      **训练阶段**：(a) Rectifier Group Training: 初始化 N=4 个 rectifier {(A_i, B_i)}, r=8 → 对每层 Mamba 模块，门控网络 G 输出 g_i → ΔW_fused = Σ g_i·(B_i@A_i) → W' = W + ΔW_fused → 伪量化 W_q' = Q_{a,b}(W') → 前向 Y = X @ W_q' → Loss = L_pixel + λ·L_feature → STE 梯度近似 → Adam(lr=1e-2) 更新所有参数，12k iters, batch=8；(b) Offline Static Routing Calibration: 冻结所有 rectifier 参数 → 对每层优化 ĝ = argmin_g L (500 iters, batch=8) → 计算每个模块最优增量 ΔW_opt[l] = Σ ĝ_i·(B_i@A_i) → 构建 SPR²Q Table。
      **推理阶段**：对每个 Mamba 模块 l：检索 ΔW_opt[l] → W_final = W + ΔW_opt[l] → 量化 W_q_final = Q_{a,b}(W_final) → 前向 Y = X @ W_q_final（计算图与原始 MambaIRv2-light 完全一致，零额外开销）。输出 Set5 PSNR=37.72 (vs FP32 38.26, 仅降 0.54dB；vs PTQ4VM +0.55dB)。
    - **系统框架**：论文未明确说明。
    - **编译框架**：论文未明确说明。
    - **kernel 调度**：使用 PaddlePaddle 框架内置量化算子，无自定义 CUDA kernel。推理时所有 rectifier 参数已离线融合，不引入额外 kernel launch 或内存访问。MambaIRv2-light 4-bit: 1.20MB (2.51× 压缩), 22.0G FLOPs (3.44× 加速)；2-bit: 1.07MB (2.81×), 18.2G (4.15×)。
    - **硬件架构**：NVIDIA RTX 4090 GPU。推理时模型结构与原始 MambaIRv2-light 完全相同，仅权重值为量化后的整数值。压缩和加速完全来自 bit-width 降低，无额外硬件修改或异构计算。
