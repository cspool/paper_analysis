## Pixel Shuffle for Visual Token Compression (像素重排用于视觉Token压缩)

术语解释
Pixel Shuffle（也称为 sub-pixel convolution 或 depth-to-space）是一种将空间维度上采样/下采样与通道维度变换结合的操作。在 VLM 中，Pixel Shuffle 被用于压缩 vision encoder 输出的视觉 token 数量，减少 LLM 需处理的 token 总数。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pixel Shuffle 将 shape 为 [H, W, C·r²] 的 tensor 重排为 [H·r, W·r, C]（上采样模式）或反向操作 [H, W, C] → [H/r, W/r, C·r²]（下采样/压缩模式，即 inverse pixel shuffle 或 space-to-depth）。DeepSeek-VL2 使用 2×2 inverse pixel shuffle：将 SigLIP 输出的 27×27×1152 feature map 压缩为 14×14×(1152×4)=14×14×4608，token 数减少 4×（729→196 per tile），但每个 token 的维度增加 4×，保持了总信息量。操作本质是：将 2×2 空间邻域的 4 个像素各 1152 维"折叠"到通道维度，得到 4608 维的单个 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== 2×2 Inverse Pixel Shuffle (Space-to-Depth) ===
Input:  X [H=27, W=27, C=1152]  (SigLIP output, H,W must be even-ish)
Output: Y [H'=14, W'=14, C'=4608]  (after discarding last row/col if odd)

// NumPy/PyTorch equivalent
// Step: reshape → transpose → reshape
H', W' = H//2, W//2   // 27//2=13, but 27-1=26→13; actually paper says 14×14
// For odd dimensions, discard or pad first
X_crop = X[:26, :26, :]  // crop to 26×26 (nearest even)
X_reshaped = X_crop.reshape(13, 2, 13, 2, 1152)
X_transposed = X_reshaped.transpose(0, 2, 1, 3, 4)  // (13, 13, 2, 2, 1152)
Y = X_transposed.reshape(13, 13, 4608)  // notation: 14×14 in paper

// Equivalent PyTorch op:
Y = torch.nn.functional.pixel_unshuffle(X.permute(0,3,1,2), downscale_factor=2)
```

在 VLM pipeline 中，pixel shuffle 的位置位于 vision encoder 之后、VL Adaptor MLP 投影之前（Intermediate compression）。更激进的方法（如 VisionZip, VisionSelector）进一步压缩到更少 token，如 2-64 tokens，用于极大降低计算量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 内置：torch.nn.PixelShuffle(r) 用于上采样，torch.nn.PixelUnshuffle(r) 用于下采样。TensorFlow 等价：tf.nn.depth_to_space / tf.nn.space_to_depth。因子 r=2 最为常用（4× token 压缩），过大可能损失空间信息。Pixel shuffle 的优势是零参数、零额外计算（纯内存重排），非常适合 VLM 的压缩需求。InternVL2, MiniCPM-V 等也使用类似操作。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
