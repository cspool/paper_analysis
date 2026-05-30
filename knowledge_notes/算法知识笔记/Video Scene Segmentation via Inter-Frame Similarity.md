## Video Scene Segmentation via Inter-Frame Similarity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Scene Segmentation via Inter-Frame Similarity 是 TDC 论文提出的基于帧间语义相似度的场景分割方法。与传统按固定时长切分（fixed-duration clips）不同，使用 DINOv2 提取每帧 768-d embedding，计算连续帧 cosine similarity，选择 S-1 个相似度最低的帧对位置作为场景边界。每个 segment 内部语义一致，segment 间语义差异最大，保证后续 TDC 压缩在时序一致的上下文内进行。最大 segment 数 S=24。消融：S=1 (不分割) MVBench 下降 9.2 点；S=48 无额外提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Scene Segmentation Pipeline
embeddings = [normalize(DINOv2(f)) for f in frames]  # T × 768-d
sims = [(i, dot(embeddings[i], embeddings[i+1])) for i in range(T-1)]
sims.sort(key=lambda x: x[1])                        # ascending similarity
split_idx = sorted([s[0] for s in sims[:S-1]])       # S-1 lowest-sim boundaries

scenes = []
prev = 0
for idx in split_idx:
    scenes.append(frames[prev:idx+1])
    prev = idx + 1
scenes.append(frames[prev:])                          # last scene
```
DINOv2 特征对光照/视角鲁棒、对内容变化敏感。低相似度点即语义变化点（场景切换、物体出现/消失）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DINOv2 加载: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')`。计算开销：T 次 ViT forward + T-1 次 cosine similarity。类似 DIG 的 CAFS 但目的不同：CAFS 选代表性帧，TDC 选分割边界。

涉及论文标题：
- Multimodal_Long_Video_Modeling_Based_on_Temporal_Dynamic_Context
