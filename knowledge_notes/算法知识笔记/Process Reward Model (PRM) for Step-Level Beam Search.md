## Process Reward Model (PRM) for Step-Level Beam Search

术语是什么？通过联网搜索让回答具体和精准。

Process Reward Model (PRM)是一种对LLM生成过程的中间步骤（而非最终结果）进行评分的reward model。与Outcome Reward Model (ORM)仅评估最终completion正确性不同，PRM为每个generation step（如数学推理中的一行推导）输出一个process score，指示该步骤是否正确或向正确答案前进。PRM使Beam Search等step-level搜索算法可以在中间步骤剪枝低质量路径，而不必等待完整生成。典型PRM如Skywork-1.5B-PRM，在每个generated token或step boundary处输出[0,1]的process score。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Step-level Beam Search with PRM:
```
Input: prompt P, model M, beam width W, max steps S, PRM
Output: best completion

beam = {empty_path}  // 每条path含generated tokens + cumulative score
for step = 1 to S:
    candidates = []
    for each path in beam:
        next_token = M.generate_next(path)  // 生成一个step的token(s)
        new_path = path + next_token
        process_score = PRM(P, new_path)   // PRM评分当前步骤
        new_path.cumulative_score *= process_score  // 累积
        candidates.append(new_path)
    beam = top_W(candidates, key=cumulative_score)  // 剪枝到W条
    if all paths in beam are complete: break
return beam[0]
```

论文在Snapdragon平台上使用Skywork-1.5B-PRM实现step-level Beam Search。在decode阶段，beam中的W条路径映射到batch_size=W，同样利用HMX的tile空行。每step结束后PRM打分并剪枝，保留top-W高质量路径。Beam Search中Qwen2.5-1.5B和Llama3.2-1B可达到与各自3B版本相当或略优的accuracy-cost效率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PRM的训练：在人工标注的process-level supervision数据上fine-tune（标注每个推理步骤的正确性），输出为[0,1]分数。使用时：(1) 确定step boundary（如每句/每行/每个逻辑段落）；(2) 在step boundary处调用PRM评分；(3) 累积分数（乘积或求和）用于路径比较和剪枝。开源PRM如Skywork-PRM系列，通过HuggingFace Transformers加载。注意PRM评分本身引入额外compute overhead（每次step boundary需额外forward pass）。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

