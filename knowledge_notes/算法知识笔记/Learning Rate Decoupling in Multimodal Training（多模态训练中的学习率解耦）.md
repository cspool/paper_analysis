## Learning Rate Decoupling in Multimodal Training（多模态训练中的学习率解耦）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learning Rate Decoupling是LMFusion提出的多模态模型训练策略。核心思想：为不同模态的参数组分配独立学习率（lr），各模态参数以不同速度更新或完全冻结。LMFusion将参数划分为文本参数组θ_text（配以η_text）和图像参数组θ_img（配以η_img），学习率比r = η_text/η_img控制文本模块相对更新速度。主实验使用r=0（文本完全冻结），消融探索r ∈ {0, 0.1, 1}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 优化器配置
optimizer = AdamW([
    {'params': θ_text, 'lr': η_text},    # η_text = 0 (冻结) or η_img/10 or η_img
    {'params': θ_img,  'lr': η_img}      # η_img = 1e-4, cosine decay
])

# 三种配置的实验结论：
# r=1 (等速/standard continual pretraining):
#   No Separation: HellaSwag -15%, persistent -7% gap
# r=0.1 (文本慢更新):
#   No Separation: gap缩小到-2%, 但image性能也下降 — 存在trade-off
# r=0 (文本冻结):
#   Deep Separation: 语言能力完全保持, image性能最佳 — Pareto最优
```

关键发现：r=0在dense模型中严重损害image learning（共享参数被冻结后image数据无法学习attention pattern），但在Deep Separation中是帕累托最优——image模块有独立可训练参数，不受文本冻结影响。此技术可推广到任何需保护预训练知识的迁移学习场景（domain adaptation、continual learning、multi-task learning）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch实现：通过`torch.optim.AdamW`的不同param_groups设置差异化lr，或通过`requires_grad=False`+分组实现。关键设计选择是"哪些参数应被冻结/慢更新"——LMFusion证明仅当模态计算路径被充分解耦（Deep Separation）后，冻结才能同时实现知识保留和新能力学习。

涉及论文标题：
- LMFusion: Adapting Pretrained Language Models for Multimodal Generation
