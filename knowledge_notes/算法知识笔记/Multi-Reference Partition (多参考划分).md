## Multi-Reference Partition (多参考划分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Reference Partition 是 Free-MoRef 中的 vision token 划分策略，将长 vision token 序列按时间关系划分为多个 short parallel reference chunks。过程：(1) 将 vision tokens 按时间分为 M 个 units；(2) 每个 unit 内沿时间分解为 N 个 fragments；(3) 聚合不同 unit 的相同 index fragment → N 个 reference chunks。参数 M 和 N 为手动配置：M 越大各 reference 间时间交集越多；M=1 时 chunk 时间完全独立。各 chunk 分配相同 system prompt 和 question。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
units = split_into_groups(vision_tokens, M)
for j in range(M):
    fragments[j] = split_into_groups(units[j], N)
for i in range(N):
    chunk_i = concat([fragments[j][i] for j in range(M)])
    parallel_inputs[i] = concat([system_prompt, chunk_i, question])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Free-MoRef 固定 M=64。128 frames: N=2；256 frames: N=4；512 frames: N=8。M 影响 sparse attention pattern：M 越小各 chunk 时间连续性越强（利于 Spatial Perception），M 越大 tokens 分布越均匀（利于 Temporal Perception）。纯 token 重组操作，无额外计算开销。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference
