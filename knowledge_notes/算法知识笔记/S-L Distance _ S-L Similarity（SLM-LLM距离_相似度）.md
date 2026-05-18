## S-L Distance / S-L Similarity（SLM-LLM距离/相似度）

术语是什么？通过联网搜索让回答具体和精准。

S-L Distance（SLM-LLM Distance）是AIMS提出的度量，表示LLM生成的一个subtask需要多少额外的SLM subtask才能产生与之相似（匹配）的输出。具体定义：对于LLM的subtask L_i，如果SLM在执行了d个额外subtask后产生的subtask S_{i+d}与L_i语义相似（SBERT cosine similarity > threshold），则L_i的S-L distance为d。若无法找到匹配的SLM subtask，S-L distance设为infinity。S-L Similarity则是在该匹配点的SBERT余弦相似度值。该度量刻画了SLM和LLM在subtask粒度上的不对齐程度——SLM通常产出更细粒度的subtask，需要多个SLM subtask才能覆盖一个LLM subtask的内容。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

S-L Distance的计算过程（基于offline profiling数据）：

```
// S-L Distance计算（offline profiling阶段）
Input: LLM subtask sequence L = [L1, L2, ..., Ln]
       SLM subtask sequence S = [S1, S2, ..., Sm] (m >= n, SLM更细粒度)
       similarity threshold κ

Output: distance array D[1..n]

for each LLM subtask L_i:
    matched = false
    for d = 0 to (m - i):  // 搜索SLM中对应位置
        if SBERT_cosine_sim(S_{i+d}, L_i) > κ:
            D[i] = d
            matched = true
            break
    if not matched:
        D[i] = INF  // 无匹配SLM subtask
```

论文实验发现（Figure 6）：匹配组（SLM和LLM最终输出一致）中，随着subtask序列推进，S-L distance逐渐增大（后期LLM subtask需要更多SLM subtask才能匹配）；非匹配组中，许多S-L distance达到infinity。例如，对于一个4-ST请求，第1个subtask平均S-L distance≈1，第4个可能达到2-3。这一观察动力了AIMS的SLE和CD组件设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

S-L Distance在AIMS中通过Distance Predictor（ModernBERT + LoRA fine-tune）在线预测：输入为当前subtask内容和sequence ID，输出为预测的S-L distance。预测的distance用于SLE组件——SP_SLM预测第(i+d)个subtask内容，与SP_LLM预测的第i个subtask内容比较similarity。若similarity > κ，则SLM执行当前及后续d个subtask。训练数据来自offline profiling中对所有请求生成SLM/LLM binary tree后提取的S-L distance labels。S-L distance使AIMS能在SLM结果与LLM不同时仍找到"追赶路径"，是区别于HybridLLM/Minions独立per-subtask决策的关键机制之一。

涉及论文标题：
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

