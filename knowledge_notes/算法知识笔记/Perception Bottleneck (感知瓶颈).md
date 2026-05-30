## Perception Bottleneck (感知瓶颈)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perception Bottleneck（感知瓶颈）是 D-CoDe 论文识别并命名的、在将图像预训练 VLM 扩展到视频领域时面临的核心挑战之一。它指：静态压缩策略（如均匀帧采样、空间平均池化）对所有内容等同处理，丢弃了在时间和空间维度上不均匀分布的关键视觉信息。具体表现为：(1) 时间维度——关键动作或事件可能集中在特定时间段，均匀采样可能完全跳过这些信息密集段；(2) 空间维度——平均池化对所有空间位置一视同仁，模糊了高信息量 token（物体边界、人脸、文本区域）和低信息量 token（纯色背景、模糊区域）的差异。论文通过 EgoSchema 5-frame 实验（Figure 2a）量化了这一瓶颈：uniform sampling + spatial average pooling 的 accuracy 显著低于无压缩 baseline，而 D-CoDe 的动态压缩不仅缩小了这一差距，甚至超越了 baseline。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Perception Bottleneck 的产生机制：
```
# === 静态压缩（Baseline）===
# 时间维度: uniform frame sampling
frames = sample_uniform(video, N=5)   # 每 T/5 帧取一帧
# 问题: 若关键动作发生在第 0.6T 到 0.7T 之间，
# 而采样点在 0.0T, 0.2T, 0.4T, 0.6T, 0.8T，
# 则关键动作仅被 0.6T 帧部分捕获，0.8T 帧错过后续

# 空间维度: spatial average pooling
for each frame:
    tokens = VisualEnc(frame)          # (H/p × W/p, d)
    compressed = AvgPool2d(tokens)     # 所有位置等同压缩
# 问题: 人脸区域（高信息）和天空背景（低信息）被同等平均，
# 导致人脸关键特征被天空"稀释"

# === 动态压缩（D-CoDe）===
# 时间: 均匀覆盖 + 多样性补充
frames_uniform = sample_uniform(video, floor(0.85*N))
frames_supp = select_diverse_frames(video \ frames_uniform, N - floor(0.85*N))
# 基于 CLIP semantic dissimilarity，补充语义不同的关键帧

# 空间: salience pruning + similarity merging
salience = ||tokens||_2               # ℓ2 norm 作为重要性代理
tokens_kept = TopK(tokens, key=salience, k=floor(0.625*M))
tokens_merged = greedy_merge(tokens_kept, threshold=0.9)
# 保留高激活 token，合并语义冗余的相似 token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Perception Bottleneck 是一个概念性术语（论文提出的问题定义），而非具体实现。D-CoDe 通过 Dynamic Compression（动态压缩）来解决这一瓶颈：时间维度用 supplementary frame selection（基于 CLIP 语义多样性的帧选择），空间维度用 salience-based pruning + cosine-similarity merging。消融实验（Table 4, EgoSchema, 15 frames）：Baseline（uniform+pooling）= 44.8% → +Dynamic Spatial Token Compression = 50.6%（+5.8%）→ +Dynamic Temporal Frame Selection = 51.8%（+1.2%）。两步分别验证了空间和时间动态压缩对缓解感知瓶颈的贡献。

涉及论文标题：
- D-CoDe__Scaling_Image-Pretrained_VLMs_to_Video_via_Dynamic_Compression_and_Question_Decomposition
