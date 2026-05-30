## Post-training Token Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-training Token Reduction是一种模型后处理效率优化技术——在不重新训练模型的情况下，通过减少推理时处理的token数量来降低计算量和内存占用。与训练时稀疏化（如structured pruning、distillation）不同，post-training方法直接应用于已训练好的checkpoint，无需访问训练数据或进行额外训练。Rethinking Token Reduction的方法属于此范畴：基于预训练Mamba模型，直接注入token reduction hook，零样本评估。优点：部署成本低（无需GPU训练集群）、即插即用（支持任何checkpoint）、可适应不同压缩率。缺点：极端压缩率下性能上限低于训练时方法（论文也指出微调可能进一步改善性能）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Post-training Pipeline:
# Phase 1: Inject hooks (offline, once)
for layer in pretrained_model.layers[layer_start::interval]:
    layer.register_hook('after_ssm', utrc_reduce)

# Phase 2: Inference (online)
output = pretrained_model(input_sequence)  # hooks auto-execute

# Contrast with training-based:
# Training-based: calibrate → fine-tune → validate → deploy
# Post-training: load_checkpoint → inject_hooks → deploy
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：(1) PyTorch hook注册在目标层的forward输出处；(2) 重要性计算基于激活统计量；(3) 排序+筛选保留top-k token；(4) 序列压缩重打包。适用场景：快速部署预训练LLM/SSM、边缘设备内存受限、多压缩率SaaS服务。局限：依赖高质量重要性度量，严重压缩时遭遇不可恢复信息损失。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
