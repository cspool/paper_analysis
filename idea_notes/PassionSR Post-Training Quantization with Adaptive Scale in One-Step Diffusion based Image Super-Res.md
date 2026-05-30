## PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

- baseline方法是什么？
  - **Baseline 方法**：现有的多步扩散模型量化方法（MaxMin、LSQ、Q-Diffusion、EfficientDM 等）直接应用于 OSEDiff（one-step diffusion SR 模型）。
  - **全栈执行例子（Baseline: OSEDiff + Q-Diffusion PTQ @ W6A6）**：
    - **算法pipeline**：Q-Diffusion 为多步扩散模型设计，依赖多时间步的校准策略（如多步噪声水平采样和跨步一致标定）来量化 UNet，但 OSDSR 仅有一个去噪步（time-step 为常数），这些多步特定技术失效。VAE 保持 FP32 未量化，占据 80%+ 计算量。CLIPEncoder 和 DAPE 等分支模块增加了标定复杂度但 Q-Diffusion 未提供针对性处理。
    - **系统框架**：PyTorch 推理，`LR → VAE_encoder(FP32) → UNet(W6A6) + text_embedding_branch(FP32) → VAE_decoder(FP32) → HR`。VAE encoder/decoder 的 1,781G MACs 全部在 FP32 执行，UNet 的 339G MACs 在 INT6 执行，DAPE 等附加模块约 141G MACs 在 FP32 执行。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。量化推理通过 PyTorch fake-quant 模拟，非真实 INT8 kernel。
    - **硬件架构**：论文未明确说明。运行在单张 GPU（推测 A100 40GB），PyTorch 2.0.1 + CUDA 11.8。
  - **Baseline 缺陷**：(i) VAE 未量化导致整体压缩率低（仅 ~17% params/ops 减少）；(ii) 多步校准策略在单步模型上不适用，甚至产生比 MaxMin 更差的结果（W6A6 时 Q-Diffusion PSNR 仅 19.75 vs MaxMin 15.55 vs PassionSR 25.15 on RealSR）；(iii) 分支模块未经处理，增加标定复杂度和不稳定性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：PassionSR 是首个面向 one-step diffusion SR 的 PTQ 方法，通过三个创新设计系统性解决上述缺陷。
  - **全栈执行例子（PassionSR W6A6 UNet-VAE 量化）**：
    - **算法pipeline**：
      - **模型简化**：将 OSEDiff 的 DAPE-CLIPEncoder 分支替换为基于空字符串 "" 预计算的 ClipEncoder 常数 embedding，模型简化为 PassionSR-FP = UNet + VAE（参数减少 27.13%, Ops 减少 6.25%, 性能持平）。这使得 VAE 和 UNet 可采用统一校准策略，且 VAE 显式纳入量化范围。
      - **LBQ（Learnable Boundary Quantizer）**：使用可训练上下界 B_l, B_u 替代固定 min/max 范围。量化过程 X_q = α·round(clamp(X,B_l,B_u)-B_l)/α) + B_l，其中 α=(B_u-B_l)/(2^N-1)。仅 B_l, B_u 可训练，通过 STE 反向传播，在极小参数开销下自适应找到最优量化区间。
      - **LET（Learnable Equivalent Transformation）**：对 Linear: W̃=s⊙W, X̃=(X-δ)⊘s, B̃=B+δW；对 Conv: 沿 channel 维相同变换；对 Attention: Q̃=Q⊘s, K̃=s⊙K。s 和 δ 为逐通道可学习参数，训练后 s 融入前层权重/归一化层，δ 融入权重/偏置，零额外推理开销。这解决了 Baseline 中激活离群值导致的量化困难——激活分布从分散/含大量离群值变为集中/友好（Fig. 7），W6A6 下 PSNR 从仅用 LBQ 的 23.15 提升至 25.40（+2.25 dB）。
      - **DQC（Distributed Quantization Calibration）**：Stage 1 冻结 LBQ、仅训练 LET；Stage 2 重新初始化 LBQ、联合训练。这解决了 Baseline PTQ 中量化参数训练不稳定的问题：DQC 使标定时间从 3.87h 降至 1.07h，GPU 显存从 40GB 降至 28GB。
      - **损失函数**：VAE encoder: L_VAE_e = ||V_qe(X_fp) - V_fpe(X_fp)||₂；VAE decoder: L_VAE_d = ||V_qd(X_q) - V_fpd(X_fp)||₂；UNet: L_UNet = ||I(Z_lq, ε_q) - I(Z_l, ε_fp)||₂，其中 I 是从 latent+噪声预测到输出 latent 的变换函数。模块级逐层标定避免跨模块误差累积。
    - **系统框架**：量化推理 `LR → VAE_encoder(INT8) → UNet(INT8) → VAE_decoder(INT8) → HR`，全链路 INT8。W8A8 PassionSR-UV 参数 238M（↓81.77%）、操作 1,060G（↓76.56%）；W6A6 参数 178M（↓86.32%）、操作 795G（↓82.42%）。对比 Baseline W6A6 UNet-only 量化：参数 246M（↓81.11%）、操作 3,689G（↓18.44%），PassionSR 将 VAE 量化纳入后操作压缩率从 18% 跃升至 82%。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：论文未明确说明。推理通过 `inference_single.py` 执行 PyTorch 模型推理，标定在 `ptq_quantize_single.py` 中实现 fake-quant 模拟。等价变换的硬件兼容性由 AWQ [22] 已确认。
    - **硬件架构**：论文未明确说明。标定 GPU（推测 A100 40GB），PyTorch 2.0.1 + CUDA 11.8。最终部署至移动设备或其他资源受限硬件为目标场景。
  - **对应关系的核心逻辑**：Baseline 因"VAE 不量化 → 压缩率低"，PassionSR 通过模型简化 + LET 统一 UNet/VAE 量化协议解决；Baseline 因"多步技术不适用单步模型 → 性能崩溃"，PassionSR 通过 LBQ 可学习边界 + OSD 专用损失函数解决；Baseline 因"激活离群值 → 量化误差大"，PassionSR 通过 LET 逐通道等效变换 + DQC 稳定两阶段标定解决。
