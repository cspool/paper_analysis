## CS-Fusion（Channel-Shuffle Fusion，通道混洗融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CS-Fusion 是 BI-DiffSR 提出的 UNet skip connection 特征融合模块。UNet 的 skip connection 需融合 encoder/decoder 特征，但直接 concatenation 导致维度翻倍（与二值化 ResBlock 维度不匹配），直接 addition 因两种特征值域差异巨大（可达数倍）导致小值域特征被遮盖。CS-Fusion 通过 channel shuffle 将两个输入特征按奇偶通道索引交叉重组为两个新特征，平衡值域后通过双分支二值卷积+加法融合。shuffle 后各特征混合了两个输入的信息，值域自然接近。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: x1, x2 ∈ [H, W, C]
# 输出: x_fused ∈ [H, W, C]

# Channel Shuffle: 奇偶交错
x1_sh = Concat(x1_odd, x2_even)     # [H,W,C], x1 奇数通道 + x2 偶数通道
x2_sh = Concat(x1_even, x2_odd)     # [H,W,C], x1 偶数通道 + x2 奇数通道

# 二值化卷积融合（维度不变）
x_fused = BI_Conv1(x1_sh) + BI_Conv2(x2_sh)  # [H,W,C]
```
消融：CS-Fusion=31.99dB vs Concat=31.08dB vs Split=29.67dB vs Add=18.89dB。channel shuffle 零参数、零计算，仅改变索引排列。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：构造索引映射后 `torch.cat` 拼接，配合 BI-Conv 完成融合。适用于任何需要融合两个值域差异大的特征并保持输出维度不变的场景。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---
