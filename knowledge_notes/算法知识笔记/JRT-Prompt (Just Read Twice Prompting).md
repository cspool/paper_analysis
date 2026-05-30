## JRT-Prompt (Just Read Twice Prompting)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
JRT-Prompt是将context(C)和question(Q)在prompt中重复两次后生成答案的ICL prompting策略：Ŷ=A(C,Q,C,Q)。动机源于SD通信复杂度分析——causal模型从左到右处理时小集合在前则存储需求最小。重复让模型在第二轮condition on完整prompt后决定存储什么，等价于展示所有数据顺序。无需模型修改或训练，off-the-shelf循环LM直接可用。16模型×6 ICL任务平均+11.0±1.3点。理论下界从Ω(max(|A|,|B|))降至Ω(min(|A|,|B|)/p)。N=32768/B=16/H100上11.9×于FA2的prefill吞吐量(sub-quadratic架构2N仍快于attention的N)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Standard ICL: A("Doc... Q?") → Ŷ          // Q前必须预测存什么
JRT-Prompt:   A("Doc... Q?... Doc... Q?") → Ŷ  // 第二轮有完整view
```
仅改prefill(context翻倍)，decode不变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过LM-Eval Harness将prompt中(C,Q)拆出并重复拼接，不做任务特定定制。对off-the-shelf循环LM(Mamba, Based, GLA, Mamba-2)直接可用。缺点：重复可能增加Repetition errors；context翻倍增加prefill计算。但sub-quadratic架构效率优势仍显著。

涉及论文标题：
- Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

---
