## Position-aware Subtask Routing（位置感知子任务路由）

术语是什么？通过联网搜索让回答具体和精准。

Position-aware Subtask Routing是AIMS提出的AI agent subtask调度策略：在决定一个subtask由本地SLM还是云端LLM处理时，不仅考虑subtask内容的难度特征，还考虑subtask在agent reasoning sequence中的位置（early/mid/late stage）。核心机制：使用自适应相似度阈值κ(i) = threshold_base + min(ID, 5) · 0.02（base=0.6），早期subtask κ较低（~0.62-0.64），允许更多SLM offload；后期subtask κ逐渐收紧至0.70，更倾向LLM以保证最终输出精度。该设计基于实验发现：LLM→SLM切换在Late stage精度损失9.53% vs Early stage仅5.25%，SLM→LLM切换在Late stage增益9.40% vs Early 5.14%。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Position-aware routing在AIMS的subtask-level decision pipeline中嵌入：

```
// AIMS的position-aware routing流程
function subtask_routing(ST_i, sequence_id):
    κ_i = threshold_base + min(sequence_id, 5) * 0.02
    
    // Step 1: SSE - 比较SP_SLM和SP_LLM预测的next subtask相似度
    next_slm = SP_SLM.predict(ST_i)
    next_llm = SP_LLM.predict(ST_i)
    sim = SBERT_cosine(next_slm, next_llm)
    
    if sim > κ_i:
        return SLM  // fast path
    
    // Step 2: SLE - 检查future SLM subtask是否match current LLM subtask
    d = DistancePredictor.predict(ST_i, sequence_id)
    future_slm = SP_SLM.predict_future(ST_i, d)
    if similarity(future_slm, SP_LLM.predict(ST_i)) > κ_i:
        return SLM  // S-L distance-based fallback
    
    // Step 3: CD - 迭代搜索未来收敛点
    for j in range(lookahead):
        slm_j = SP_SLM.forward_predict(ST_i, j)
        llm_j = SP_LLM.forward_predict(ST_i, j)
        if similarity(slm_j, llm_j) > κ_{i+j}:
            convergence_point = j
    if convergence_point found:
        return SLM until convergence_point
    
    // Step 4: SD - 分解为子subtask，全部通过SSE才走SLM
    sub_subtasks = SD.decompose(ST_i)
    if all(SSE(st) > κ for st in sub_subtasks):
        return SLM for all sub_subtasks
    return LLM
```

其中κ_i随着sequence_id增长而升高，构成position-aware的核心——早期宽松捕获SLM机会，后期严格保护精度。threshold_base=0.6和step=0.02通过图4a的实验数据拟合得到，论文指出各SLM-LLM pair可在其offline profiling中重新calibrate。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Position-aware routing的核心依赖是subtask位置信息（sequence ID），这在agent workflow中天然可得（每次LLM invocation对应一个sequence position）。实现要点：(1) offline profiling阶段测量不同position的SLM/LLM switching精度影响，拟合threshold曲线；(2) 在线阶段subtask predictor（SP_SLM/SP_LLM）基于Qwen3-0.6B+LoRA fine-tune，预测后续subtask内容用于similarity比较；(3) threshold可针对不同模型对和不同精度要求调整base值和step值。AIMS的position-aware routing使HybridLLM（独立per-subtask routing，无位置感知）的accuracy提升27.5%（relative），SLM usage提升31.4%。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments
