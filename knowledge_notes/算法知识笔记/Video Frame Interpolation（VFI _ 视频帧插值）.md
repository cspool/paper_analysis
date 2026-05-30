## Video Frame Interpolation（VFI / 视频帧插值）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Frame Interpolation（VFI，视频帧插值）是视频增强的核心任务之一，目标是根据两帧或多帧连续输入，合成出位于它们时间位置之间的中间帧，从而提升视频的时间分辨率（帧率）。VFI 的关键技术挑战在于：(1) 准确估计帧间的光流/运动轨迹，尤其是大运动、遮挡区域和非线性运动；(2) 合成逼真的中间帧纹理，避免伪影和模糊。VFI 广泛应用于慢动作视频生成、视频帧率上转换、视频压缩等场景。

从早期 CNN-based 方法（如 DVF、SuperSloMo、DAIN）依赖光流估计和 warp+合成 pipeline，发展到 Transformer-based 方法（如 EMA-VFI、BiFormer）通过自注意力直接建模长程时序依赖，显著提升了大运动场景下的插值质量。PMQ-VE 使用 EMA-VFI [Zhong et al. 2024] 作为 VFI 任务的 backbone 进行量化实验，EMA-VFI 通过提取运动和外观信息解决速度模糊性问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VFI 模型（以 EMA-VFI 为例）的典型 pipeline：

```
输入: I_0, I_1 ∈ R^{H×W×3}  (连续两帧)
目标: 输出 I_t (t ∈ (0,1) 时刻的中间帧)

# Step 1: 特征提取
F_0 = Encoder(I_0)           # 多尺度特征金字塔
F_1 = Encoder(I_1)

# Step 2: 运动和外观信息提取（EMA-VFI 核心）
motion_feat = MotionExtractor(F_0, F_1)    # 帧间注意力建模运动
appearance_feat = AppearanceExtractor(F_0, F_1)  # 外观信息聚合

# Step 3: 帧合成
warped_feat = BackwardWarp(F_0, F_1, motion_feat, t)
I_t = Decoder(warped_feat, appearance_feat)
```

PMQ-VE 量化 VFI 模型时，对 Encoder/Decoder 中的所有 Linear 和 MatMul 层执行逐帧量化（BMFQ → PMTD）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常用 VFI 开源实现：EMA-VFI（ECCV 2024, https://github.com/zhongzhihang/EMA-VFI）、RIFE（实时 VFI）、AMT（大规模运动 VFI）。评估数据集：Vimeo-90K（标准 VFI benchmark）、UCF101、SNU-FILM。评估指标：PSNR、SSIM、LPIPS。PMQ-VE 在 Vimeo-90K 测试集上评估，对 EMA-VFI [T] 和 [D] 两种变体分别测试 4-bit 量化性能。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
