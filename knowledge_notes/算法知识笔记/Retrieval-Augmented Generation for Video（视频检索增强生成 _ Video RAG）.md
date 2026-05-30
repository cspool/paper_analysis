## Retrieval-Augmented Generation for Video（视频检索增强生成 / Video RAG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Retrieval-Augmented Generation (RAG) 是一种将信息检索（Retrieval）与生成模型（Generation）结合的框架。在视频场景中，Video RAG 将长视频切分为多个片段（clips），对每个片段提取文本描述、视觉特征或结构化信息，构建可检索的知识库。当用户提出查询时，系统先检索与查询最相关的视频片段，再将检索到的片段作为上下文输入 LVLM 生成最终答案。标准 Video RAG pipeline 包含三个阶段：(1) Indexing：将原始视频数据组织为可检索知识库——对每个 clip 调用 LVLM 生成 text description 或用 CLIP/ViT 提取 visual feature embedding；(2) Retrieval：计算 query embedding 与各 clip embedding 的 cosine similarity，返回 Top-N 最相似的 clips；(3) Generation：将检索到的 clips（video frames + subtitles）拼接后输入 LVLM 生成回答。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NaïveRAG for video 的标准 pipeline：
```
# === Indexing (offline) ===
clips = split_video(video, fps=1.0, clip_size=64)
index = []
for clip in clips:
    desc = LVLM.describe(clip)           # text description
    emb = text_encoder.encode(desc)      # BGE/CLIP text embedding
    index.append({"clip": clip, "desc": desc, "emb": emb})

# === Retrieval (online) ===
q_emb = text_encoder.encode(query)       # query embedding
scores = [cosine_sim(q_emb, item.emb) for item in index]
top_k = argsort(scores)[:N]              # Top-N clips

# === Generation (online) ===
context = concat([index[i].clip for i in top_k])
answer = LVLM.generate(query, context=context)
```
Vgent 论文中实现的 NaïveRAG baseline（遵循 GoldFish 风格）即为上述流程——每个 video clip 作为独立 plain text document 处理，检索相似 clip 后直接将视觉 frames 输入 LVLM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Video RAG 的实现通常基于开源 LVLM（如 Qwen2.5-VL, LLaVA-Video, InternVL2.5）搭配 embedding model（如 BGE, CLIP, BERT）进行检索。在 Vgent 中，NaïveRAG 使用 BAAI/bge-large-en-v1.5 进行 embedding 计算，检索 Top-N=20 个 clips 输入 LVLM 生成回答。但 NaïveRAG 的局限在于：将每个 clip 视为独立 document，破坏了跨 clip 的时序依赖和实体连续性；此外，检索到的 clips 中存在大量 hard negatives（语义相似但与问题无关的 clip），干扰 LVLM 推理。Vgent 实验显示 NaïveRAG 在 MLVU 上反而比 base model 直接推理低 3.4 个百分点（65.4 vs 68.8），验证了 NaïveRAG 在复杂长视频任务中的失效。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding
