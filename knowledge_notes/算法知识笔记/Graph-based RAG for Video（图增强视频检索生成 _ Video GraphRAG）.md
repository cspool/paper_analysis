## Graph-based RAG for Video（图增强视频检索生成 / Video GraphRAG）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Graph-based RAG (GraphRAG) 是 RAG 的一个变体，用图结构（而非 flat index）组织检索知识库。在视频场景中，GraphRAG 将每个 video clip 建模为图的节点（node），通过共享的语义实体（entities——人物、物体、场景、动作）在节点间建立边（edge），形成视频知识图谱 G=(V, E)。这种图表示的关键优势：(1) 保留跨 clip 的语义关系和时序依赖——同一实体在不同 clip 中出现时通过共享 entity 节点关联；(2) 支持基于实体的精准检索——query keyword 匹配 entity 后直接溯源到所有相关 clip 节点；(3) 图构建是 query-independent 的离线操作，同一视频的多个问题复用同一张图，无需重复处理视频。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Vgent 的 GraphRAG pipeline（graph construction + graph-based retrieval）：
```
# === Phase 1: Graph Construction (offline) ===
G = Graph(vertices=empty, edges=empty)
U = set()  # global unique entities with descriptions

for i, clip in enumerate(clips):
    # Step 1: Entity extraction
    entities = LVLM.extract_entities(clip, subtitle)
    # entities = [{"name": "laptop", "desc": "silver laptop on desk"}, ...]
    
    # Step 2: Entity merging via BGE embedding similarity
    for e in entities:
        e_emb = BGE.encode(e.desc)
        sims = {u: cosine_sim(e_emb, BGE.encode(u.desc)) for u in U}
        if max(sims) > tau:  # tau = 0.7
            u_star = argmax(sims)
            merge(e, u_star)  # unify semantically equivalent entities
            add_edges(v_i, get_nodes(u_star))  # connect to nodes sharing entity
        else:
            U.add(e)  # new unique entity
    
    G.add_vertex(v_i)  # clip i as graph node

# === Phase 2: Graph-based Retrieval (online) ===
keywords = LVLM.extract_keywords(query)
R = set()
for k in keywords:
    for u in U:
        if cosine_sim(BGE.encode(k), BGE.encode(u.desc)) > theta:  # theta = 0.5
            R = R.union(get_nodes(u))

# Re-rank by avg similarity across all clip info (entities, descriptions, subtitles)
R_sorted = rank_by_avg_similarity(R, query_keywords)
R_topK = R_sorted[:20]
```
核心计算：entity merging 和 keyword-entity matching 均基于 BGE text embedding 的 cosine similarity——entity description 经 BGE 编码为 1024-d 向量后进行匹配。图的结构使得检索可以沿着 entity→node 的边直接找到所有相关 clips，而非遍历整个 index。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GraphRAG for video 作为 training-free pipeline 包裹任意开源 LVLM。离线阶段：视频以 1.0 FPS 采样，每 64 帧一个 clip，对每个 clip 调用 LVLM 提取 entities/actions/scenes 的 JSON——无需额外 object detection 或 OCR 模型。BGE embedding 用于 entity 合并 (tau=0.7) 和 keyword-entity 匹配 (theta=0.5)。Vgent 实验证实 GraphRAG 比 NaiveRAG 平均提升 2.9%，在 MLVU 上提升 4.1%——尤其是在 Count/Order 等多 clip 时序推理任务上提升显著（Count: 从 41.7→58.7）。但 GraphRAG 单独使用时提升有限——论文发现 44% 的 failure 案例中正确 clip 已在检索集内，噪声仍干扰 LVLM，这促使了 Structured Reasoning 后检索步骤的引入。与 Video-RAG（依赖 CLIP keyframe selection + external object detection/OCR）和 proprietary LLM-based 方法（VideoAgent, DrVideo 依赖 GPT-4 API）不同，GraphRAG 仅使用开源 LVLM + embedding model。

涉及论文标题：
- Vgent__Graph-based_Retrieval-Reasoning-Augmented_Generation_For_Long_Video_Understanding
