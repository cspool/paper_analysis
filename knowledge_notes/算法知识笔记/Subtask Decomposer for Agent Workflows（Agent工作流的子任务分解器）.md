## Subtask Decomposer for Agent Workflows（Agent工作流的子任务分解器）

术语是什么？通过联网搜索让回答具体和精准。

Subtask Decomposer (SD)是AIMS中的一个关键recovery组件，当一个复杂subtask通过SSE/SLE/CD都无法安全分配给SLM时，SD将其拆解为更简单、更细粒度的子subtask序列，使SLM能够逐步处理。SD基于Qwen3-0.6B + LoRA fine-tune，输入为当前subtask内容和SP_LLM预测的next subtask（作为分解目标），输出为分解后的子subtask序列{SSP_1, SSP_2, ..., SSP_m}。之后AIMS对每个子subtask重跑SSE，仅当所有子subtask都通过SSE（即SLM和LLM对每个子subtask的next prediction相似）时才将整组分配给SLM；否则将原始subtask交给LLM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Subtask Decomposer的决策流程：

```
// Subtask Decomposer工作流
function subtask_decomposer_route(ST_i):
    // 用SD模型将复杂subtask分解为子subtask序列
    next_llm = SP_LLM.predict(ST_i)
    sub_subtasks = SD.decompose(ST_i, next_llm)
    // sub_subtasks = {SST_1, SST_2, ..., SST_m}
    
    // 逐个子subtask评估LLM-SLM一致性
    for each SST_j in sub_subtasks:
        next_slm_j = SP_SLM.predict(SST_j)
        next_llm_j = SP_LLM.predict(SST_j)
        sim_j = SBERT_cosine(next_slm_j, next_llm_j)
        
        if sim_j < κ_position(j):
            // 任一子subtask不通过 → 原始subtask走LLM
            return LLM
    
    // 全部通过 → 整组给SLM
    for each SST_j in sub_subtasks:
        SLM.process(SST_j)
    return combined_result
```

论文示例：HotpotQA中subtask "Verify Shirley Cameron's father, including corroborating biographical details, to confirm his identity as James Cameron's maternal grandfather" → SD分解为：SST1="Search for Shirley Cameron's father" → SST2="Extract father's full name" → SST3="Find key biographical details for verification (e.g., birth/death dates)" → SST4="Confirm and state the maternal grandfather's name"。四个子subtask比原始复杂subtask更容易通过SSE检查。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SD的训练数据来自offline profiling：对请求的subtask binary tree中SLM生成的更细粒度subtask序列与LLM的对应subtask做配对，从中学习"如何将一个LLM-level subtask分解为SLM-level子subtask序列"。SD采用整组分配策略（全部分解子subtask适合SLM才offload整组，否则走LLM）以避免增加LLM调用次数——若逐个子subtask独立路由，可能部分走SLM部分走LLM，反而增加总LLM调用。实验消融：移除SD使SLM usage下降5.54%、accuracy下降1.58%。SD的局限：(1) 保守策略可能错失混合case（部分子subtask可SLM）；(2) 依赖SP_SLM/SP_LLM的预测准确性。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

