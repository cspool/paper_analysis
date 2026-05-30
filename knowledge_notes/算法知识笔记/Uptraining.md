## Uptraining

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Uptraining 由 Ainslie et al. (2023) 首次使用，指在修改模型架构后继续语言建模训练——区别于 fine-tuning（通常在不同数据集上继续训练）。在 SUPRA 中，uptraining 具体指：在 softmax Transformer attention 层中添加 MLP kernel 参数后，在相同预训练语料（RefinedWeb）上继续训练约 5% 原始 tokens，同时更新新增参数和原有参数。关键洞察：不追求近似 softmax（T2R 策略），而是直接替换 attention 让模型适应新范式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Uptraining vs Fine-tuning vs Pre-training:
- Pre-training: 随机初始化 → 完整训练 (1-8T tokens)
- Uptraining:  预训练模型 → 修改架构 → 继续训练 (5% tokens, 同数据集)
- Fine-tuning: 预训练模型 → 新数据集/任务 → 少量训练

SUPRA uptraining: Mistral-7B (8T tokens) → 添加 MLP kernel → RefinedWeb 100B tokens
LR: 3e-5→1e-5 cosine, 1000步 warmup, Adam (β1=0.9, β2=0.95), seq=2048, H100+FSDP
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Uptraining 的收益来自 base model 的质量：Mistral-SUPRA avg 64.0 vs Llama2-SUPRA avg 58.6（更高质量预训练数据带来持续优势）。Uptraining 需要低于预训练的 LR 以保持已学知识。局限性：Instruct-tuned 模型线性化效果差于 base model；继承 base model 的 biases。

涉及论文标题：
- Linearizing_Large_Language_Models

---
