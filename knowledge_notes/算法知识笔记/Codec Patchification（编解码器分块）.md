## Codec Patchification（编解码器分块）

术语是什么？
Codec Patchification 是 OneVision-Encoder 提出的编解码器引导的视觉 token 输入构造策略。核心思想：HEVC/H.265 编解码器天然将视频分解为"稳定空间上下文"（I-frame 全量编码）和"稀疏时序更新"（P-frame 仅运动补偿残差），这种分解揭示了判别性视觉证据（"surprise"）仅稀疏存在于运动/变化区域。Codec Patchification 利用 HEVC 暴露的运动矢量（motion vectors）和预测残差（prediction residuals）作为 patch 级信息熵代理，仅在密集视频帧（64帧）中选择 3.1%–25% 最显著 patch 送入 ViT。包含三种形式：Dense Video-Codec Patchification（GOP结构，I-frame全量 + P-frame稀疏）、Chunk-wise Patchification（均匀分块+单帧采样）、Single-Image Spatial Patchification（静态图像行主序 patchify）。

从算法pipeline角度拆解术语：
Codec Patchification 在 ViT 编码前的数据预处理阶段运作：

```
# 64帧视频, GOP=32, token budget B=2048
# Step1: HEVC解码提取motion vectors + residuals (CPU)
for each GOP: decode I-frame(RGB) + extract mv,res per P-frame

# Step2: Patch级显著性评分
for each P-frame patch(y,x):
    saliency = sum(||mv||₂ over patch) + sum(|res| over patch)

# Step3: 全局Top-K (跨所有P-frames)
selected = topk(all_P_saliency, k=B-512)  # 512=2个I-frame全量
tokens = concat(I_patches (512), P_patches[selected] (1536))

# Step4: 3D-RoPE + ViT encoding + attentive pooling
tokens = tokens + 3D_RoPE(sparse_positions)
features = ViT(tokens) → attentive_pooling(features)
```

训练时三种模式混合：Codec 50%, Frame Sampling 37.5%, Tiling 12.5%。Token budget 固定在 clip level（非 per-GOP），确保全局最优分配。推理可灵活切换 Codec 稀疏或传统帧采样。

术语一般如何实现？如何使用？
使用 FFmpeg/libx265 提取 motion vectors 和 residuals（不重编码，直接从原始 bitstream 提取）。显著性评分在 CPU 上计算，选中的 patch indices 通过 visible_indices 传给 ViT，未选中 patch 不存储/不计算/不传梯度。Token 压缩：64帧×256 patches=16384 → 2048=87.5% reduction。限制：需 HEVC 格式存储；无法处理实时流的未来帧。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence
