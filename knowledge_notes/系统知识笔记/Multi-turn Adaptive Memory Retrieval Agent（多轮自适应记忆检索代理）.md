## Multi-turn Adaptive Memory Retrieval Agent（多轮自适应记忆检索代理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-turn Adaptive Memory Retrieval Agent是WorldMM中负责多模态记忆检索的LLM驱动调度模块。与传统的"固定k条检索"不同，该Agent在每轮迭代中自主决定：(1) 选哪个记忆类型（episodic/semantic/visual），(2) 用什么搜索query，(3) 是否已收集足够信息（输出STOP停止）。每次决策基于原始用户query和所有历史检索记录。最多N=5轮。Retrieval Agent与Response Agent分离设计，使检索专注信息收集、响应专注答案生成。多轮设计的关键优势：Agent可以根据前几轮检索结果调整策略——例如先episodic获取时间戳，再visual按时间戳获取帧图像。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Retrieval Agent的决策与执行流程：
```
# 系统组件：RetrievalAgent(LLM), MemoryStore(三记忆), ResponseAgent(LLM)
query = "What were we doing last time we discussed AC temperature?"
history = []
for round in 1..5:
    # LLM推理: 根据query+历史决定下一步
    decision = LLM(prompt=Fig.16, query=query, history=history)
    if decision["decision"] == "answer": break  # → ResponseAgent
    
    m_type = decision["selected memory"]["memory type"]
    query_i = decision["selected memory"]["search query"]
    
    if m_type == "episodic":
        results = episodic_search(query_i)  # PPR + cross-scale rerank
    elif m_type == "semantic":
        results = semantic_search(query_i)  # PPR edge-score
    elif m_type == "visual":
        if is_timestamp_range(query_i):
            results = frame_lookup(query_i)  # M_v^I直接索引
        else:
            results = visual_feature_search(query_i)  # cosine sim
    history.append({"memory": m_type, "query": query_i, "results": results})

# 输出: 全部history → ResponseAgent生成答案
answer = ResponseAgent(query, history)
```
Round-by-round示例（论文Table 16）：R1→Episodic,"discussing AC temp"（定位但未确认活动）→R2→Episodic,"air conditioning"（扩大搜索得更多片段）→R3→Visual,"DAY2 18:34:01-18:34:29"（获取帧确认hot pot）→R4→Answer(A)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Retrieval Agent使用LLM（GPT-5或Qwen3-VL-8B）推理，prompt模板见Fig.16（含决策模式search/answer、记忆类型选择、查询格式规范、few-shot示例）。输出为JSON格式（decision + selected memory）。最大迭代次数N=5与baseline Ego-R1和M3-Agent保持一致。EgoLifeQA上5轮vs1轮提升9.3%。系统延迟方面：WorldMM在延迟-精度trade-off上优于long video LLMs和RAG方法（论文Fig.6）。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning
