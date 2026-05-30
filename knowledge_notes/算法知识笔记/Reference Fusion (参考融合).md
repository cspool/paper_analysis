## Reference Fusion (参考融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Reference Fusion 是在 LLM decoder 中间层执行的 vision token 剪枝与合并操作。基于 FastV 观察（vision tokens 在 shallow layers 均匀贡献，deep layers 集中于 question tokens）。流程：(1) 在第 L 层，基于 A ∈ R^{N × l_ques × l_vis}；(2) 沿 l_ques 平均 → E ∈ R^{N × l_vis}；(3) 每个 chunk 保留 top 1/N tokens（剪枝 1-1/N）；(4) 按时间关系聚合为 global reference；(5) system prompt/question 直接拷贝；(6) 后续 layers 仅用 global reference 做标准 attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for i in range(N):
    E_i = mean(A_i, dim=l_ques)
    kept_tokens_i = vision_chunk_i[argsort(E_i, desc)[:l_vis_i//N]]
global_vision = temporal_merge(kept_tokens_all)
global_seq = concat([system_prompt, global_vision, question])
# layers L+1..end: standard causal attention on global_seq
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Free-MoRef 配置：128 frames → L=3 (drop 50%)，256 frames → L=6 (drop 75%)。过早 fusion (L=1) 导致信息丢失（65.4 vs L=3 的 66.3）。双重作用：减少 deep layers 计算量 + 补偿 shallow layers 缺失的跨 chunk vision 交互。消融：仅 MoRef Attention 无 Fusion 时 Overall 65.8 vs 含 Fusion 66.3。

涉及论文标题：
- Free-MoRef__Instantly_Multiplexing_Context_Perception_Capabilities_of_Video-MLLMs_within_Single_Inference
