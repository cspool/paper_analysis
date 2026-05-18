## Best-of-N (Outcome Reward Model-based Sampling)

术语是什么？通过联网搜索让回答具体和精准。

Best-of-N是一种parallel test-time scaling方法，在decode阶段生成N条完整的候选completion，然后使用Outcome Reward Model (ORM)或外部verifier对每条候选打分，选择得分最高的作为最终输出。与Beam Search在中间步骤剪枝不同，Best-of-N仅在生成结束后做一次性选择。在数学推理等可通过自动验证检查答案正确性的任务中特别有效。Best-of-N的compute cost与N成正比（每条候选独立生成），但N条候选的decode可以在一个batch中并行执行，利用hardware的batch parallelism。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: prompt P, model M, generation budget N, reward model RM
Output: best completion C*

candidates = []
for i = 1 to N:
    C_i = M.generate(P, temperature=T)  // 独立采样N条
    candidates.append(C_i)

for each C_i in candidates:
    score_i = RM(P, C_i)   // ORM评分，如Skywork-PRM
C* = argmax(score_i)
return C*
```

在Mobile NPU上，Best-of-N的N条候选映射到decode batch，填充HMX 32×32 tile的N行（vs 单路径decode仅1行有效）。论文实测：Qwen2.5-1.5B在MATH500上，N=16时accuracy可达~70%（超越3B base model ~60%），同时per-token decode latency仅轻微增加（HMX计算时间几乎不随batch增大而增加）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Best-of-N的实现要点：(1) Reward model选择：数学推理可用outcome verification（答案匹配），也可用训练好的ORM如Skywork-1.5B-PRM；(2) Temperature设置：需要非零temperature以产生多样化候选；(3) Batch parallelization：将N条候选的decode组织为batch_size=N的推理，利用hardware tile-level parallelism；(4) 在移动NPU上，lm_head/logits常保留在CPU（32-bit虚拟地址空间限制），batch增大时CPU logits计算可能成为瓶颈（B=16时>50%时间）。开源实现见llama.cpp-npu。

涉及论文标题：
- Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

