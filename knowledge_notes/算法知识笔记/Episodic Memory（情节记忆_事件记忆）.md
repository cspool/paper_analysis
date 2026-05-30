## Episodic Memory（情节记忆/事件记忆）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Episodic Memory（情节记忆）是受认知心理学启发的记忆概念，在视频理解中指存储具体事件/片段发生的"事实性"文本记忆。在WorldMM中，Episodic Memory以多时间尺度的知识图谱集合形式构建：将长视频按不同时间分辨率T={t₀,t₁,...,t_N}（如30s, 3min, 10min, 1h）分别切分，每段生成caption后提取(entity, action, entity)三元组，为每个时间尺度构建独立的知识图谱G_{t_i}，最终形成记忆集合M_e={G_{t₀},...,G_{t_N}}。与Semantic Memory不同，Episodic Memory存储具体事件（"某天18:34分Shure把空调设到26度"），而非长期关系或习惯。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Episodic Memory构建与检索流程：
```
# === 构建阶段 ===
输入: 视频V, 时间尺度集合T={30s, 3min, 10min, 1h}
M_e = {}
for t_i in T:
    segments = split_video(V, duration=t_i)
    G_ti = empty_KG()
    for seg in segments:
        frames = sample_frames(seg, fps=0.5)
        transcript = Whisper.transcribe(seg)
        caption = VideoLLM(frames, transcript)
        triplets = LLM.extract_triplets(caption)  # [(e1,action,e2)]
        for (subj, pred, obj) in triplets:
            G_ti.add_edge(subj, obj, relation=pred)
    M_e[t_i] = G_ti

# === 检索阶段 (Coarse-to-Fine) ===
candidates = []
for t_i in T:
    ppr = PersonalizedPageRank(G_ti, seed=extract_entities(q))
    top_k_nodes = argsort(ppr.scores)[:k]
    candidates += [(t_i, G_ti.get_captions(node)) for node in top_k_nodes]
top_m = LLM.cross_scale_rerank(q, candidates)  # prompt见Fig.13
return top_m
```
关键计算：PPR迭代 s=α·A^T·s+(1-α)·s₀，收敛后s[i]为节点i的PPR分数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖Video LLM生成caption + LLM提取三元组（prompt模板见WorldMM Fig.10-11）+ HippoRAG的PPR检索框架。多时间尺度根据视频总长设定：周级别{30s,3m,10m,1h}，小时级{10s,30s,3m,10m}。固定单尺度替代多尺度在WorldMM消融中导致6.1%精度下降。相关概念也出现在HERMES（同样区分episodic/semantic memory）和EgoRAG（层级事件记忆但仅单尺度）中。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning
- HERMES__temporal-coHERent_long-forM_understanding_with_Episodes_and_Semantics
