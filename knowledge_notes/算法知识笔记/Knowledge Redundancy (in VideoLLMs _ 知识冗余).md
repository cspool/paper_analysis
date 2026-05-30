## Knowledge Redundancy (in VideoLLMs / 知识冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Knowledge Redundancy（知识冗余）在 VideoLLM 推理上下文中指：LLM 的多层自注意力机制内在包含的 token 级冗余信息——许多 visual tokens 即使被丢弃也不会显著影响模型对视频内容的理解，因为 LLM 的高层多模态知识可以推断出被丢弃 token 承载的信息。这与 Temporal Redundancy（时间冗余，相邻帧的像素级视觉重复）形成对比：时间冗余是"低层"的（low-level，基于帧间像素/特征距离），知识冗余是"高层"的（high-level，基于 LLM 学到的语义理解）。RETAKE 的核心洞察是：时间冗余压缩虽然计算开销低但信息损失大（仅基于帧间距离），知识冗余压缩虽然需要额外计算（需要 LLM 前向计算 attention）但信息保持更好，两者联合可取得最优的压缩比-精度 trade-off。知识冗余概念源自 H2O（Heavy-Hitter Oracle）等 LLM token pruning 工作，这些工作发现 attention scores 可以预测 token 重要性——低 attention 的 KV cache token 可被安全丢弃。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 RETAKE 中，知识冗余通过 PivotKV 模块在 LLM 内部捕获和利用：
```
# 知识冗余的量化: 通过 attention 权重衡量 token 重要性
# 低 attention token = 高知识冗余 = 可安全压缩

# 在 LLM 第 l 层的 self-attention 中:
A = Softmax(QK^T / sqrt(d_h))

# token j 的重要性 = 所有 query positions 对它的总关注度
importance[j] = sum_{all queries} mean_{all heads} A[:, j]

# 高 importance -> 低知识冗余 -> 保留
# 低 importance -> 高知识冗余 -> 丢弃

# 关键设计: pivot frames 的 token 不论 attention 高低都保留
# 这确保低层时空细节（temporal structure）不丢失
```
trade-off 分析实验：固定总压缩比 0.25，变化 alpha_dp/alpha_kv 比例。alpha_dp/alpha_kv 越高 = 更依赖知识冗余压缩。最优比例在 2~3 之间，表明适度偏好知识冗余策略。但继续增大 alpha_dp/alpha_kv 会增加 FLOPs（因为 DPSelect 压缩少意味着更多 token 进入 LLM 计算）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
知识冗余的利用不需要额外训练——RETAKE 直接复用 VideoLLM 已有的注意力权重（在 chunked prefilling 过程中自然计算得到）。实际使用时，用户设置总 compression budget 和 alpha_dp/alpha_kv 比例，DPSelect 先做帧级粗筛（低时间冗余），PivotKV 再做 token 级精剪（低知识冗余）。该方法对 Needle QA（需要精确定位单个关键帧中细微信息）略有精度损失（~1%），但对 Action Order、Key Information Retrieval、Temporal Grounding 等粗粒度任务反有提升，因为去除冗余信息增加了有效信息密度。适用于任何基于 Transformer attention 的 VideoLLM，无需模型修改。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding
