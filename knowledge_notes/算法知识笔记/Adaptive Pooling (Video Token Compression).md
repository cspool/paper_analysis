## Adaptive Pooling (Video Token Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Pooling（自适应池化）在视频语言模型中是一种非参数的视觉 token 压缩方法，通过 PyTorch 的 AdaptiveAvgPool3d 将任意长度的视频帧序列压缩为固定数量的视觉 token。典型实现（PLLaVA, Xu et al. 2024）：给定 n 帧视频经 ViT 编码后得到形状为 [n, H, W, D] 的特征张量，通过 AdaptiveAvgPool3d(target_shape=(T', H', W')) 池化到固定 token 数，使 LLM 输入 token 数恒定，无论原始视频多长。PLLaVA 使用 AdaptiveAvgPool3d 将视频特征池化到 [16, 12, 12] tokens（16 帧 × 144 = 2304 tokens per frame → pooled to 16×12×12）。此方法的优点：(1) 零参数，无需训练；(2) GPU 显存恒定；(3) 简单高效。缺点：(1) 平均池化丢失显著视觉细节（尤其是空间定位信息和细小物体）；(2) 当原始帧数 >> 目标帧数时，大量信息被平均化稀释；(3) 对指令内容无感知——池化与用户问题无关，可能丢弃问题关键帧。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Adaptive Pooling 在 PLLaVA 中的使用：

```
# === PLLaVA Adaptive Pooling Pipeline ===
# 输入: n frames (可变), ViT-L/14 encoder
# 输出: 固定 token 数的特征

# Step 1: ViT 编码
features = []
for frame in video_frames:  # n frames
    feat = ViT(frame)  # [257, 1024] (1 CLS + 256 patches)
    features.append(feat[1:])  # 丢弃 CLS, [256, 1024]
features = stack(features)  # [n, 256, 1024]

# Step 2: Reshape to 3D
features_3d = features.reshape(n, 16, 16, 1024)  # H=W=sqrt(256)=16

# Step 3: Adaptive 3D Pooling
# 目标: [T', H', W'] → 如 [4, 12, 12] 或 [16, 12, 12]
pooled = AdaptiveAvgPool3d(features_3d, target_size=(T', 12, 12))
# PLLaVA 训练: (16, 12, 12) → 16*12*12 = 2304 tokens
# PLLaVA 推理 (32帧): (16, 12, 12) → 仍 2304 tokens (从32帧池化)

# Step 4: 送入 Projector → LLM
tokens = Projector(pooled.reshape(16*12*12, 1024))
answer = LLM(Concat(tokens, text_tokens))
```

VideoLLaMB 的消融比较（Table 8）：
- Adaptive Pooling (PLLaVA): EgoSchema 45.6%
- Mean Pooling (uniform, 同 VideoLLaMB 设置): 51.61%
- VideoLLaMB: 53.8%
- Adaptive Pooling 甚至不如 Mean Pooling，可能因训练-推理不一致（训练时固定帧数池化与推理时自适应帧数池化的分布不匹配）

改进方向：PPLLaVA (ICLR 2026) 将 AdaptiveAvgPool3d 扩展为 Prompt-Guided Pooling——引入 CLIP 文本-视觉对齐模块计算 token 级 relevancy map，使用 relevancy 作为 3D 卷积权重进行加权池化，实现 18× token compression 同时保持 SOTA 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Adaptive Pooling 在 PyTorch 中通过 `torch.nn.AdaptiveAvgPool3d(output_size)` 实现，output_size 可以是元组 (T, H, W) 或 int（等维度）。在视频理解中，output_size 的选择是关键设计决策：较小的 output_size 减少 LLM 计算量但丢失更多信息；较大的 output_size 保留信息但增加计算成本。PLLaVA 使用 (16, 12, 12)=2304 tokens 作为默认值。Adaptive Pooling 也被 LLaVA-NeXT-Video（使用 position extrapolation + sampling）和其他 video LLM 用作 baseline。它的核心局限性——"对指令无感知的均匀压缩"——驱动了后续如 PPLLaVA（prompt-guided pooling）、VideoLLaMB（recurrent memory）等方法的提出。

涉及论文标题：
- VideoLLaMB__Long-context_Video_Understanding_with_Recurrent_Memory_Bridges
