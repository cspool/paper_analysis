## CP-Down / CP-Up（Consistent-Pixel-Downsample / Upsample，一致性像素下采样/上采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CP-Down 和 CP-Up 是 BI-DiffSR 提出的二值化友好 UNet 特征缩放模块。传统 UNet 使用 stride 卷积或 pooling 改变特征分辨率，导致输入输出维度不匹配，identity shortcut（二值化模型中传递全精度信息的关键通道）无法使用。CP-Down/CP-Up 将所有维度变化操作隔离到独立的缩放模块中，确保主模块（ResBlock）维度始终一致。CP-Down：先将输入沿通道均分为两组 → 各组经（二值化）卷积处理（维度不变，可加 shortcut）→ 两组结果相加 → Pixel-UnShuffle 降低分辨率并增加通道数。CP-Up：输入经两个（二值化）卷积处理 → 输出沿通道拼接 → Pixel-Shuffle 提升分辨率并减少通道数。核心公式：CP-Down `x_out = PixelUnshuffle(C1(x_s1) + C2(x_s2))`，CP-Up `x_out = PixelShuffle(Concat(C1(x_in), C2(x_in)))`。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CP-Down 伪代码（[H,W,C] → [H/2,W/2,2C]）：
```
x_s1, x_s2 = Chunk(x, 2, dim=C)    # [H,W,C/2] × 2
x1 = BI_Conv1(x_s1)                 # 二值化卷积, 维度不变
x2 = BI_Conv2(x_s2)                 # 二值化卷积, 维度不变
x_mid = x1 + x2                     # 加法融合, 可用shortcut
x_out = PixelUnshuffle(x_mid, 2)    # [H/2,W/2,2C]
```
CP-Up 伪代码（[H,W,C] → [2H,2W,C/2]）：
```
x1 = BI_Conv1(x)                    # [H,W,C]
x2 = BI_Conv2(x)                    # [H,W,C]
x_cat = Concat(x1, x2, dim=C)       # [H,W,2C]
x_out = PixelShuffle(x_cat, 2)      # [2H,2W,C/2]
```
消融验证：+CP-Down&CP-Up 使 PSNR 从 29.29dB（仅 identity shortcut）提升至 31.08dB (+1.79dB)。不引入额外参数——输入通道被均分后各自处理，总 MACs 不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 原生实现：`torch.nn.PixelUnshuffle(2)` 和 `torch.nn.PixelShuffle(2)`（v1.9+）。使用 Pixel-(Un)Shuffle 而非 stride/transposed convolution 的好处：(1) 无可学习参数，减少二值化训练负担；(2) 空间↔通道变换可精确反转，不引入额外信息损失。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---
