## Spatio-Temporal Video Super-Resolution (STVSR / 时空视频超分辨率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Spatio-Temporal Video Super-Resolution（STVSR，时空视频超分辨率）是视频增强中最具挑战性的任务之一，需要同时提升视频的空间分辨率（空间维度+）和时间分辨率（帧率，时间维度+）。也就是说，输入一个低分辨率、低帧率的视频序列，输出高分辨率、高帧率的视频序列。STVSR 结合了 VSR（空间维度超分）和 VFI（时间维度插帧）两个子问题，但两问题互相耦合——更好的空间细节有助于准确的帧间运动估计，反之更精确的运动信息也有助于多帧空间信息聚合。

PMQ-VE 使用 RSTT [Geng et al. 2022] 作为 STVSR backbone。RSTT 是一个实时 Transformer-based STVSR 模型，通过从编码器不同层构建特征字典（feature dictionary），在解码阶段反复查询该字典来同时增强空间和时间分辨率。其优势在于实时性（满足实际部署需求），但高计算量也使其成为量化的理想候选。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

STVSR（以 RSTT 为例）的 pipeline：

```
输入: {LR_1, LR_2, ..., LR_7}  (7 帧低分辨率、低帧率)
目标: 输出 {HR_t1, HR_t2, ..., HR_tM}  (高分辨率、高帧率序列)

# Step 1: 多级编码
F_level1, F_level2, F_level3 = PyramidEncoder(LR_frames)
# 构建多级特征字典
Dict = StackDict(F_level1, F_level2, F_level3)

# Step 2: 字典查询解码
for each output frame t:
    # 从特征字典中查询对应时空位置信息
    Q_t = LearnableQuery(t)              # 可学习的位置编码查询
    F_t = CrossAttn(Q_t, Dict)           # 跨注意力查询字典
    # 多层解码
    for level in [1, 2, 3]:
        F_t = DecoderBlock(F_t, Dict[level])
    HR_t = ReconstructionHead(F_t)       # 空间上采样+帧合成
```

RSTT 中所有 Linear 层（Q/K/V 投影、FFN 的 FC1/FC2）和 MatMul 层（Q@K^T、Attn@V）均被 PMQ-VE 量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

常用 STVSR 方法：Zooming Slow-Mo（首个端到端 STVSR）、TMNet（时空调制网络）、RSTT（实时 Transformer STVSR, CVPR 2022）、STDAN（可变形注意力 STVSR）。评估数据集：Vimeo-90K（含 Vimeo-Fast、Vimeo-Medium、Vimeo-Slow 三个子集，按运动速度分级）、Vid4。评估指标：PSNR、SSIM（Y 通道）。PMQ-VE 在以上四个 benchmark 上均达到最优 4-bit 和 2-bit 量化性能。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
