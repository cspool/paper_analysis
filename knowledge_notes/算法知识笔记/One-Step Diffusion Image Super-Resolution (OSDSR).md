## One-Step Diffusion Image Super-Resolution (OSDSR)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-Step Diffusion Image Super-Resolution (OSDSR) 是将扩散模型的多步去噪过程压缩为单步推理的图像超分范式。传统扩散 SR 模型（如 StableSR、DiffBIR、SeeSR）需要 T = 50-1000 步迭代去噪，每步需完整运行 UNet 推理，高延迟阻碍实际部署。OSD 模型通过 score distillation 或 consistency distillation 技术将去噪步数减至 1：对给定的低分辨率输入 LR，模型在单次前向传播中直接从噪声/低质量 latent 恢复到高质量 latent Z_h，再经 VAE decoder 生成高分辨率输出 HR。代表性 OSD SR 模型包括 SinSR（CVPR 2024）、OSEDiff（NeurIPS 2024）、DFOSD（2024）。以 OSEDiff 为例，其结构包含 UNet（去噪预测）、VAE（编解码）、DAPE（detail-aware prompt encoder）和 CLIPEncoder（文本条件），FP32 下参数 1,303M、操作 4,523G。OSD 模型虽然步数降至 1，但单步计算量极大（UNet+VEA MACs 超 2,100G），亟需量化压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OSEDiff 的 one-step 推理流程（FP32）：
```
# 输入: LR image, 输出: HR image (×4 upscaling)
Z_l = VAE_encoder(LR)                          # latent encoding, ~1,781G MACs
ε_pred = UNet(Z_l, text_embedding)              # noise prediction, ~339G MACs (1 step)
Z_h = sqrt(1/α̂) * Z_l - sqrt((1-α̂)/α̂) * ε_pred  # one-step latent recovery
HR = VAE_decoder(Z_h)                           # latent decoding
```
与多步扩散对比：多步模型需将 T=50 步的 UNet 推理串联，总 MACs = T × UNet_MACs + VAE_MACs ≈ 50 × 339G + 1,781G ≈ 18,731G。OSD 仅 1 步 UNet，总 MACs ≈ 2,262G（含 DAPE），加速 ~8.3×。但 VAE 的 1,781G MACs（78.8%）成为主要瓶颈——这是 PassionSR 选择同时量化 VAE 而非仅量化 UNet 的根本原因。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
OSD SR 模型的实现基于预训练扩散模型（如 Stable Diffusion 2.1）的 score distillation：(1) 使用预训练 SD 的 UNet + VAE 权重初始化；(2) 通过 score distillation loss 将多步教师的行为蒸馏到单步学生模型；(3) 在真实世界 SR 数据集上微调。OSEDiff 开源代码（https://github.com/cswry/OSEDiff）提供完整训练和推理流程。PassionSR 直接使用 OSEDiff 作为量化 backbone 并进一步简化：将 DAPE-CLIPEncoder 分支替换为基于空字符串预计算的常数 embedding（参数从 1,303M 降至 949M，性能持平），得到仅含 UNet+VAE 的 PassionSR-FP 模型，便于统一量化标定。

涉及论文标题：
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

---
