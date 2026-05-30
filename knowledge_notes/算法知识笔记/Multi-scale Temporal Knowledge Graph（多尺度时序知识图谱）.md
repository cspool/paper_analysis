## Multi-scale Temporal Knowledge Graph（多尺度时序知识图谱）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-scale Temporal Knowledge Graph是WorldMM中Episodic Memory的核心数据结构——对同一视频按不同时间粒度分别构建多个知识图谱。时间粒度T={t₀<t₁<...<t_N}，如周级视频{30s,3min,10min,1h}，小时级{10s,30s,3m,10m}。每个G_{t_i}是从粒度t_i的caption提取(entity,action,entity)三元组构成的图，实体为节点、动作为边。多尺度设计动机：不同query需不同时间跨度——"Where did I leave my glasses?"需秒级，"What happened in the match second half?"需十分钟级。单尺度要么冗余太多，要么信息不足。检索时从所有尺度并行召回后用LLM cross-scale reranker动态选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
T = {30s, 3min, 10min, 1h}  # EgoLifeQA配置
for t_i in T:
    for seg in partition(V, t_i):
        cap = VideoLLM.caption(seg)
        triplets = LLM.extract(cap)
    G_ti = build_graph(triplets)

# 检索: coarse-to-fine
all_candidates = []
for t_i in T:
    ppr = PPR(G_ti, seed=query_entities)
    all_candidates += [(t_i, cap) for cap in top_k(ppr)]
# LLM联合评估选择: 具体事件→偏好细粒度; 习惯/关系→偏好粗粒度
top_m = LLM.rerank(query, all_candidates)
```
粗粒度caption由LLM通过合并、摘要提示构造（prompt Fig.12）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
时间尺度选择根据视频总长设定。扰动实验({20s/2m/5m/50m}→65.2%, {30s/3m/10m/1h}→65.6%, {1m/5m/15m/1.5h}→64.8%)证明对精确值鲁棒，收益来自多尺度设计本身。固定单尺度导致6.1%精度下降。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning
