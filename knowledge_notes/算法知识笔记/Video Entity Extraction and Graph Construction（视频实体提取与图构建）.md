## Video Entity Extraction and Graph Construction（视频实体提取与图构建）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Video Entity Extraction and Graph Construction 是 Vgent 中的离线预处理阶段，将原始长视频转换为可检索的结构化知识图谱。该过程分为两个子步骤：(1) Visual Entity Extraction——对每个 video clip（64 帧），调用 LVLM 提取关键语义实体（entities：物体、人物、场景）、动作（actions：交互/行为描述）和场景（scenes：地点/环境），输出为结构化 JSON。该步骤同时利用视频的视觉内容（frames）和口语内容（subtitles/ASR），形成图文对齐的实体描述。(2) Graph Construction——基于提取的实体构建视频知识图谱 G=(V, E)，通过 BGE text embedding 的 cosine similarity 识别和合并跨 clip 的语义等价实体，在共享实体的节点间建立边。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
实体提取的 JSON 输出示例（来自论文 Appendix C.1）：
```
{
  "entities": [
    {"entity name": "sailboat", "description": "A classic sailboat with white sails and wooden rigging"},
    {"entity name": "man", "description": "A man wearing a dark sweater"},
    {"entity name": "ocean", "description": "A calm ocean under a partly cloudy sky"}
  ],
  "actions": [
    {"entity name": "sailboat", "description": "sailing smoothly on the water"},
    {"entity name": "man", "description": "steering the sailboat"}
  ],
  "scenes": [
    {"location": "open sea"}
  ]
}
```
实体合并算法：
```
U = set()  # global unique entities
for clip i, entity e_j in extracted entities:
    t_j = BGE.encode(e_j.description)     # 1024-d text embedding
    scores = {u: cosine_sim(t_j, BGE.encode(u.description)) for u in U}
    u_star = argmax(scores)
    if scores[u_star] > 0.7:               # tau = 0.7 merging threshold
        merge(e_j, u_star)                 # unify as same entity
        add_edges(v_i, {v | u_star in entities(v)})
    else:
        U.add(e_j)                         # new distinct entity
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 采样率：1.0 FPS，每 64 帧一个 clip。(2) LVLM 用于实体提取的 prompt 设计（Appendix B.1）：要求 LVLM 以 JSON 格式输出 entities、actions 和 scenes，每个 entity 附带 description 字段供 embedding 编码。第一人称视频中 subject 被描述为 "me"。(3) Embedding 模型：BAAI/bge-large-en-v1.5 (1024-d) 在实验中优于 CLIP 和 BERT——BGE 是专门为语义文本相似度优化的 embedding 模型。(4) 合并阈值 tau=0.7：较高阈值偏向精准匹配，避免浅层语义相似导致的错误合并。(5) 图构建的离线性：图构建是最耗时的步骤（20.13 sec/min-video），但这是 query-independent 的一次性开销——同一视频的多个问题复用同一张图，在多问题场景下摊薄开销。Vgent 在每视频 3 个问题的 VideoMME 上实现 1.73x 加速。(6) 当前局限：仅使用 textual entity descriptions 构建图，未包含 visual embeddings 或 frame-level features——论文在 Limitations 中指出这是未来改进方向。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding
