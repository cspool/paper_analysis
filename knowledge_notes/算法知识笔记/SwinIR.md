## SwinIR

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwinIR（Shifted Windows Image Restoration）是由 ETH Zurich 提出的基于 Swin Transformer 架构的图像复原模型。其核心架构包含三个模块：(1) Shallow Feature Extraction——一个 3×3 卷积层将输入 RGB 图像映射到高维特征空间；(2) Deep Feature Extraction——K 个 Residual Swin Transformer Block (RSTB) 堆叠加末尾 3×3 卷积提取深层特征；(3) High-Quality Image Reconstruction——融合浅层和深层特征后，通过 sub-pixel convolution（pixel shuffle）上采样重建超分辨率图像。SwinIR 使用残差学习（预测 LQ→HQ 的残差而非 HQ 本身）和长跳跃连接。与传统 CNN-based SR 模型相比，SwinIR 在更少参数下获得更好性能，得益于 Swin Transformer 的自注意力机制和 shifted window 机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SwinIR-light（2DQuant 使用的版本，4 RSTB, 6 STL/RSTB, 6 heads, embed_dim=60, window=8, MLP ratio=2）的前向计算：
```
# Input: I_LQ (H×W×3)
F_0 = Conv3x3(I_LQ)                  # 浅层特征 (H×W×C)
F = F_0
for k in 1..4:                        # 4 个 RSTB
    F_in = F
    for l in 1..6:                    # 6 个 STL
        # Window-based MSA
        X = WindowPartition(LayerNorm(F))   # (HW/M^2, M^2, C)
        Q,K,V = X@W_q, X@W_k, X@W_v         # Linear 投影 → INT 量化对象
        Attn = SoftMax(Q@K^T/√d + B)        # Batch MatMul → INT 量化对象
        X = Attn@V                            # Batch MatMul
        F = F + WindowReverse(X)
        # MLP
        F = F + FC2(GELU(FC1(LayerNorm(F))))  # FC1, FC2 → INT 量化对象
    F = Conv3x3(F) + F_in              # RSTB 残差连接
F_DF = Conv3x3(F)                      # 深层特征
I_RHQ = SubPixelConv(F_0 + F_DF)      # 重建 (rH×rW×3)
return I_RHQ + Upsample(I_LQ)          # 残差学习
```
2DQuant 对其中所有 Linear（FC1, FC2, Q/K/V 投影）和 Batch MatMul（Q@K^T, Attn@V）执行伪量化和 INT 算术转换，这覆盖了总 FLOPs 的 85.66%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SwinIR 的官方实现基于 PyTorch，开源仓库为 https://github.com/JingyunLiang/SwinIR。模型有多个尺度变体：light（4 RSTB）、classical（6 RSTB）、large（更多通道）。训练使用 DF2K（DIV2K+Flickr2K）数据集，测试在 Set5/Set14/B100/Urban100/Manga109 上评估 PSNR/SSIM。预训练模型可通过 Google Drive 下载。在 2DQuant 的使用场景中，SwinIR-light 被作为 baseline 模型进行 PTQ 量化，其紧凑的参数规模（3.42MB）和高效结构使其适合边缘部署。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution

---
