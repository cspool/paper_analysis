## Video Super-Resolution (VSR / 视频超分辨率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video Super-Resolution（VSR，视频超分辨率）是从低分辨率（LR）视频序列中利用多帧时序信息重建高分辨率（HR）视频的任务。与单图超分（SISR）不同，VSR 的核心优势在于跨帧聚合亚像素信息——相邻帧中同一物体可能存在微小的亚像素位移，合理利用这些信息可以从多帧低分辨率观测中恢复出超过单帧物理限制的高频细节。

VSR 的历史演进：早期方法使用显式光流估计+运动补偿+融合（如 VESPCN、TOFlow），后演进为可变形卷积对齐（EDVR、BasicVSR），再到最近的 Transformer-based 方法（PSRT、MIA）通过自注意力机制隐式建模时序依赖。PMQ-VE 使用 MIA [Zhou et al. 2024] 作为 VSR backbone 进行量化实验。MIA 采用 masked intra-frame attention 和 inter-frame attention 块，通过掩码注意力更好利用之前增强帧的特征。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

VSR（以 MIA 为例）的典型 pipeline：

```
输入: {LR_t, LR_{t-1}, ..., LR_{t-N+1}}  (N 帧低分辨率序列)
目标: 输出 HR_t (当前时刻的高分辨率帧)

# Step 1: 特征传播与对齐
F_{t-1} = propagated_feature_from_previous_timestep
F_aligned = IntraFrameAttention(LR_t) + InterFrameAttention(F_{t-1}, LR_t)
# MIA 使用 masked attention: 仅关注最相关的帧间/帧内区域

# Step 2: 特征增强
F_enhanced = TransformerBlocks(F_aligned)  # 多层自注意力+FFN

# Step 3: 上采样重建
HR_t = Upsample(F_enhanced) + BicubicUpsample(LR_t)  # 残差学习
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常用 VSR 开源实现：BasicVSR/BasicVSR++（https://github.com/ckkelvinchan/BasicVSR）、MIA（CVPR 2024）、PSRT（NeurIPS 2022）。评估数据集：Vimeo-90K（主要用于 VSR 的训练和评估）、Vid4（经典 VSR benchmark，4 个视频片段）、REDS、UDM10。评估指标：PSNR、SSIM（Y 通道）。PMQ-VE 在 Vimeo-90K 和 Vid4 上评估 MIA 的 4-bit 量化性能，报告 PSNR/SSIM。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
