## Token Weighting for Multimodal SFT Loss (多模态SFT损失加权)

术语解释
对VLM SFT阶段不同任务类型的loss token施加差异化权重，解决长输出样本（如4000+ tokens video caption）主导loss导致短输出任务（如MCQ仅1 token）性能退化的问题。Molmo2采用固定权重（caption=0.1, pointing=0.2）+ sqrt inverse weighting（其他任务 weight=4/√n）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLM SFT数据mix包含输出长度差异极大的任务。一个4000-token caption sample对总loss的贡献是1-token MCQ的4000倍。即使降低sampling rate，其token count仍主导梯度。Token Weighting在per-token CE loss上乘weight：(1) video caption weight=0.1；(2) pointing weight=0.2；(3) 其他任务 weight=4/√n_answer_tokens。n=1 (MCQ)→4.0, n=100→0.4, n=400→0.2。单个sample total weighted loss增长从O(n)降为O(√n)。Molmo2 Table 8b: 去除token weighting导致QA avg -0.8但Caption F1 +0.5——验证QA↔Caption trade-off。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
if task == 'video_caption': weight = 0.1
elif task == 'pointing':    weight = 0.2
else:                       weight = 4.0 / sqrt(n_answer_tokens)
weighted_loss = token_loss * loss_mask * weight  # Per-token weighting
# Gradient averaged over global mean tokens across all devices (not per-device)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Per-device gradient除以全局平均loss token数（非per-device own count），避免短样本隐式up-weight。适用场景：任何含输出长度差异大的多任务VLM SFT训练。

涉及论文标题：
- Molmo2__Open_Weights_and_Data_for_Vision-Language_Models_with_Video_Understanding_and_Grounding
