## MQAR (Multi-Query Associative Recall)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Arora et al. (2023) 提出的合成任务：上下文中多个 key-value pair，给定 query 需正确回忆对应 value。难度随序列长度和 pair 数量增加。与 in-context learning 能力相关（Elhage 2021, Olsson 2022），已成为评估新架构设计的重要基准。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Input: color->blue, animal->dog, color->  → Target: blue
```
模型需选择性检索 "color→blue" 而非其他 pair。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Finch 在 MQAR 上极高准确率，超越所有已知训练过大模型的非 Transformer 架构。GoldFinch作为hybrid架构（后1/3层为GOLD attention），在MQAR上取得完美分数（100% recall），与纯attention Transformer持平，验证了hybrid设计在保持linear attention效率的同时不牺牲长程精确检索能力。

涉及论文标题：
- Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence
- GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

---
