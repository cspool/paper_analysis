## Concept-aware Retrieval Algorithm（概念感知检索算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Concept-aware Retrieval Algorithm 是 PEARL 的检索核心，用于在流式视频场景中精确检索与用户查询相关的个性化概念信息和历史视觉证据。四步流程：(1) **概念检索**——从查询 Q 中提取概念名，按键从 Concept Memory 检索 Csub；(2) **查询重写（Query Rewriting）**——将 Q 中概念名替换为对应文本描述，生成 Q̃（如 "What is Adaliz doing?" → "What is a young female with long black hair and oval face doing?"）；(3) **流式记忆检索**——用 Qwen3-VL-Embedding-2B 编码 Q̃ 为 e^Q，与 Streaming Memory 中所有 clip 嵌入 {ei}i≤tq 计算余弦相似度，选 Top-K；(4) **时序上下文扩展**——对每个选中 clip 扩展其相邻 N 个 clips 以捕获局部时序上下文。最终将 Csub、Vcontext、当前 clip X^tq 和 Q 送入 VLM 生成答案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
检索算法的具体计算流程：

```
def concept_aware_retrieval(Q, t_q, ConceptMemory, StreamingMemory, 
                             embedding_model, vlm, K=4, N=1):
    # Phase 1: Concept Retrieval
    mentioned = [n for n in ConceptMemory if n in Q]
    C_sub = {n: ConceptMemory[n] for n in mentioned}
    rules = {n: e["description"] for n, e in C_sub.items()}
    
    # Phase 2: Query Rewriting
    Q_tilde = Q
    for name, desc in rules.items():
        Q_tilde = Q_tilde.replace(name, desc)
    
    # Phase 3: Streaming Memory Retrieval
    e_Q = embedding_model.encode(Q_tilde)          # [d_embed]
    sims = [(i, cosine_sim(e_Q, e_i)) 
            for i, (_, e_i) in enumerate(StreamingMemory) 
            if timestamp(X_i) <= t_q]
    top_K = sorted(sims, key=lambda x: x[1], reverse=True)[:K]
    
    # Phase 4: Temporal Adjacent Expansion
    V_context = set()
    for idx, _ in top_K:
        for offset in range(-N, N+1):
            if 0 <= idx + offset < len(StreamingMemory):
                V_context.add(StreamingMemory[idx + offset].clip)
    
    return C_sub, V_context
```

超参数分析（Fig.4）：K=0 时无法检索历史证据，准确率极低；K≥3 后性能趋于饱和。N=1 相比 N=0 有显著提升（捕获时序上下文），N=2 增量收益有限（噪声抵消）。默认 K=4, N=1。消融实验（Table 4）显示完整 pipeline 相比无 Query Rewriting 版本提升 4.28% Avg。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 PEARL 代码库的 `video_qa_inference.py`。使用 Qwen3-VL-Embedding-2B（MRL 训练，支持 64-2048 维嵌入）作为嵌入模型，余弦相似度通过标准向量内积实现，Query Rewriting 通过 VLM 纯文本 prompt 完成。延迟分解（Fig.5）显示核心检索和重写模块延迟极低且恒定，主要瓶颈仍是 LLM 推理。适用场景：任何需要将个性化概念与流式视频检索结合的多模态应用。

涉及论文标题：
- PEARL__Personalized_Streaming_Video_Understanding_Model
