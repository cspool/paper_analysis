## Post-Training Quantization (PTQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Training Quantization（PTQ，后训练量化）是一种在模型完成常规浮点训练后将预训练模型的高精度参数（FP32/FP16）转换为低比特整数表示（INT8/INT4/INT2等）的模型压缩技术。其核心流程为：(1) 加载已训练好的 FP32 模型权重；(2) 使用少量校准数据（通常无需标签）统计各层权重和激活的数值分布；(3) 为每层确定量化参数（scale factor s, zero point z, clipping bounds [l, u]）；(4) 执行量化：将浮点值 v 映射为离散整数值 v_int = round((clip(v,l,u) - l) / s)，其中 s = (u-l)/(2^N-1)；(5) 推理时使用整数算术替代浮点运算，实现存储压缩和计算加速。与 QAT 的核心区别在于 PTQ 不修改模型权重本身（仅确定量化器的 clip bounds），无需重新训练，只需分钟级别校准时间，适合训练资源受限或快速部署场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 2DQuant 论文的 PTQ pipeline 为例，量化一个 Transformer-based SR 模型的流程如下：
```
# Stage 1: 离线量化参数搜索
for each layer in model:
    w = layer.weight  # FP32 权重
    a = calibrate_activations(layer, calibration_images)
    if is_symmetric(w):  # 对称钟形分布
        l_best, u_best = symmetric_mse_search(w, K=100)
    else:  # 非对称指数分布
        l_best, u_best = asymmetric_mse_search(a, K=100)
    quantizer[layer] = (l_best, u_best)

# Stage 2: 蒸馏微调
for iter in range(3000):
    x = next_batch(calibration_set)
    o_fp = fp_model(x)           # 教师输出 (FP32)
    o_q = quantized_model(x)     # 学生输出 (INT4 算术)
    loss = L1(o_fp, o_q) + λ * feature_L2(f_fp, f_q)
    loss.backward()  # STE 通过量化操作回传梯度
    update_clip_bounds()  # 仅更新 l, u，不更新权重

# 部署推理
convert_all_linear_to_INT_arithmetic(model)
```
量化后的 Linear 层计算：`y = INT_MATMUL(W_int, x_int) * s_w * s_x`，其中 `s_w`, `s_x` 为 scale factors。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PTQ 的通用实现方式：(1) 使用 PyTorch 或 TensorFlow 的量化 API（如 torch.quantization、torch.fake_quantize）；(2) 通过校准数据统计 min/max 或使用 MSE/熵最小化搜索 clip bounds；(3) 可选使用 AdaRound、GPTQ、AWQ 等高级方法优化 rounding 策略；(4) 将模型导出为量化格式（如 ONNX QInt8、TensorRT INT8 engine）。在 PyTorch 中，基本用法为：`torch.quantization.prepare(model, inplace=True)` → 校准 → `torch.quantization.convert(model, inplace=True)`。现代 PTQ 方法（如 GPTQ、AWQ）可直接通过 pip 包使用，支持 HuggingFace 模型的 one-shot 量化。AffineQuant 进一步将 PTQ 的等价变换从缩放/平移扩展到完整的仿射变换（d² 维优化空间），通过 Gradual Mask 保持矩阵可逆性。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- AFPQ Asymmetric Floating Point Quantization for LLMs
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- AffineQuant Affine Transformation Quantization for Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs
- Bridging the Gap Between Promise and Performance for FP4 Quantization
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization
- DMQ Dissecting Outliers of Diffusion Models for Post-Training Quantization
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement
- PT²-LLM Post-Training Ternarization for Large Language Models
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference
- PassionSR Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution
- Q-VDiT Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
- QT-DoG Quantization-Aware Training for Domain Generalization
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
- Training Dynamics Impact Post-Training Quantization Robustness

S²Q-VDiT 将 PTQ 范式扩展到视频扩散模型（V-DMs）领域。V-DMs 的 token 序列极长（n=s×t vs I-DMs 的 n=s），导致两个关键挑战：(1) 校准预算受限（仅几十样本 vs I-DMs 的数千样本）下随机采样方差极大；(2) 长序列中所有 token 均等处理的量化损失优化效率低。S²Q-VDiT 提出 Hessian-aware Salient Data Selection (SDS)——联合扩散信息量 C_diff = ||x_t-x_{t-1}||²/||x_t||² 和量化敏感度 C_quant = ||x_t^T x_t||_2（Levenberg-Marquardt Hessian 近似）的乘积得分筛选校准数据——和 Attention-guided Sparse Token Distillation (STD)——通过多头注意力图 A 计算 token-wise 重要性权重 λ_j，重加权量化损失 L_quant = (1/n)Σλ_j||θ^f(x_j)-θ^q(x_j)||²。在 CogVideoX-2B/5B 和 HunyuanVideo-13B 上 W4A6 几乎无损、W4A4 维持 95% 性能，首次探索 V-DMs 的 4-bit 激活量化。部署基于 ViDiT-Q 和 FlatQuant 的 CUDA kernel，CogVideoX-5B 实现 3.94× 模型压缩、1.56× 推理显存节省、1.28× 推理加速。代码: https://github.com/wlfeng0509/s2q-vdit。

SynQ (ICLR 2025) 在 PTQ 场景下与 Genie (Jeon et al., 2023b) 集成：SynQ 的低通滤波、CAM 对齐和困难样本软标签策略应用于 Genie 生成的合成数据集，在 ResNet-18 ImageNet W2A2/W2A4/W3A3/W4A4 上平均提升 0.66%p 准确率，证明其 synthesis-aware fine-tuning 兼容 PTQ 范式下的 clipping/rounding 优化。

Q-resafe 对 PTQ 的安全评估：Q-resafe (Chen et al., ICML 2025) 首次系统评估了主流 PTQ 方法（AWQ、AQLM）在多种校准数据集和不同位宽下的安全影响。关键发现：(1) 所有 PTQ 方法都会损害安全——INT4 下 ASR 从 FP16 的 0.3% 升至 18.5-42.4%；(2) 有微调的 PTQ（AQLM）在 benign 数据集上更安全（ASR=18.5%），但在有害数据集上风险急剧放大（ASR=77.4%）；(3) 无微调的 PTQ（AWQ）安全退化中等（ASR=42.4%），但无法通过数据集选择控制风险；(4) 低 bit-width 显著加剧安全退化——2-bit 下 QLoRA ASR=82.0%。Q-resafe 提出的安全修补方法可在不损效用前提下将 PTQ 后 ASR 恢复至接近 FP16 水平。

PassionSR 将 PTQ 范式首次扩展到 one-step diffusion (OSD) 图像超分模型。其 PTQ 方法的核心差异在于：(1) 同时量化 UNet 和 VAE（传统多步扩散量化仅量化 UNet），因为 OSD 模型中去噪步数减至 1 后 VAE 占据 80%+ 计算量；(2) 通过可学习边界量化器（LBQ）以梯度下降替代传统 fixed-range 量化搜索；(3) 通过等效变换（LET）抑制激活离群值后量化；(4) 采用两阶段分布式标定（DQC）稳定训练，以 PTQ 级效率（1.07h 标定时间）实现接近 QAT 的量化精度。W8A8 下参数压缩 81.77%、操作减少 76.56%，W6A6 下参数压缩 86.32%、操作减少 82.42%。

ParoQuant 在 W4A16 PTQ 中引入了 Scaled Pairwise Rotation——结合 channel-wise scaling 和独立 Givens 旋转——通过 AdamW 梯度下降优化旋转角度和缩放因子（而非 AWQ 的 grid search），并在推理时通过 fused CUDA kernel 应用逆变换（~10% 开销）。两阶段逐层优化可学习变换参数，专门针对推理 LLM 长链思维生成中的误差累积问题。

PT²-LLM 将 PTQ 范式扩展到三值量化（Ternarization）领域。传统 PTQ 方法处理 2-8 bit 均匀量化，而 PT²-LLM 实现了仅 3 个量化级别（{−1,0,+1}，等效 1.58-bit）的 post-training 场景。其 ATQ 通过两阶段无训练优化（ITF 交替优化 + AGA 激活感知对齐）将三值参数在无梯度反传的情况下精细优化，是 PTQ 范式在极端低比特三值场景的首次系统探索。

QT-DoG 从域泛化（DG）角度揭示了 PTQ 的关键局限：PTQ (OBC, Frantar et al. 2022) 在 PACS 上无法提升 DG 性能（OBC 83.7% vs 无量化 ERM 84.7%），因为 PTQ 仅做推理时压缩而无训练过程，无法通过量化噪声引导模型搜索平坦极小值。这验证了 QT-DoG 的核心主张——只有 QAT（训练时注入量化噪声）才能通过 Hessian 交互机制找到平坦极小值，PTQ 的单纯推理时量化不具备此正则化效果。

Q-VDiT 将 PTQ 范式首次扩展到视频 Diffusion Transformer (V-DiT) 领域。视频 DiT 相比图像 DiT 有更高的 token 信息密度（n=s×t vs n=s），直接应用图像量化方法的 PTQ 导致剧烈性能退化。Q-VDiT 的 PTQ 创新包括：(1) TQE (Token-aware Quantization Estimator)——利用 H(Δ)≤H(W) 的信息论性质，在 token 和 feature 维度使用 rank=1 低秩参数估计和补偿量化误差，参数开销仅 d_in+d_out+t；(2) TMD (Temporal Maintenance Distillation)——在传统 MSE 重建损失上增加帧间时序分布 KL 散度对齐项，使每帧优化受整体视频分布共同引导。W3A6 下 Scene Consistency 翻倍（12.04→23.40），W4A6 几乎无损。校准用时 12.5-12.9 小时（W8A8），推理时通过 LoRunner Kernel 融合 TQE 低秩分支，额外延迟<5%。
