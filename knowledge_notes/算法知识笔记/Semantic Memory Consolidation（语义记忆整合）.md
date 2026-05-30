## Semantic Memory Consolidation（语义记忆整合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic Memory Consolidation是WorldMM中维护持续演化的长期语义知识图谱的增量更新机制。捕获跨场景的抽象知识——人际关系、行为习惯、偏好等，区别于Episodic Memory的具体事件存储。Consolidation过程：新视频段到达时，先用embedding相似度(c >0.6)检测新三元组与已有图谱的重叠/冲突，再交LLM判断哪些旧三元组应删除(T_remove)、哪些应新建或修改(T_update)，执行G_new=(G_old\T_remove)∪T_update。避免纯append式记忆膨胀和冲突信息共存（如"dislikes sweet food" vs "likes sweet desserts"）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Consolidation流程：
```
输入: 当前语义图G_s, 新语义三元组T_new
for triplet_new in T_new:
    e_new = embed(triplet_new)
    matches = [(t_old, sim) for t_old in G_s 
               if cosine(embed(t_old), e_new) > 0.6]
    if matches:
        # LLM决策 (prompt Fig.15)
        result = LLM.consolidate(triplet_new, matches)
        G_s = (G_s - result["triples to remove"]) ∪ {result["updated triple"]}
    else:
        G_s.add(triplet_new)
```
实例：新"[I, uses WeChat for, money transfers]"与已有"[I, uses WeChat to send money]"合并；冲突"[Lucia, dislikes, overly sweet food]"替代"[Lucia, likes, sweet desserts]"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用LLM执行合并决策，embedding模型编码三元组文本。检索用PPR边级评分(edge_score=ppr(u)+ppr(v))取top-10三元组。去除Consolidation在HabitInsight类别上导致约7%精度下降。灵感来自认知科学中的记忆巩固理论。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning
