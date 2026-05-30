## Catastrophic Forgetting in Multimodal MoE Fine-tuning (多模态 MoE 微调中的灾难性遗忘)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Catastrophic Forgetting（灾难性遗忘）是指神经网络在学会新任务后，显著丧失在旧任务上已获得能力。在多模态 LLM 的上下文中，指 LLM 在全参数微调学习视觉理解等新模态时，遗忘了原有的文本理解、推理、代码、数学等能力。MoExtend 揭示了 **MoE 架构对全参数微调导致的遗忘比 dense LLM 更敏感**——这是因为 MoE 的 expert 专业化特性使每个 expert 存储更集中的知识块，全参数更新时被整体覆盖。

从算法pipeline角度拆解术语：

**全参数微调导致 MoE 遗忘的机制：**
```
# 原始 Mixtral 8x7B：8 experts/层 × 32 layers
# 全参数微调时所有 expert FFN 参数更新
for epoch in training_epochs:
    for batch in multimodal_data:  # 视觉+文本指令数据混合
        for x in concatenated_tokens:
            indices, weights = router(x)  
            loss = cross_entropy(pred, target)
            # 所有被选中的 expert FFN 参数都参与更新
            W_gate -= lr * ∂loss/∂W_gate   # ← 文本知识所在权重被覆盖
            W_up   -= lr * ∂loss/∂W_up     # ← 同前
            W_down -= lr * ∂loss/∂W_down   # ← 同前
```

**遗忘程度对比（基于 Mixtral 8x7B，来自 MoExtend Table 2）：**
```
方法              | Avg. drop ↓（7 个文本 benchmark 平均）
LLaVA-1.5-7B     | -0.81（dense LLM，遗忘轻微）
LLaVA-1.5-13B    | -0.27（dense LLM，遗忘轻微）
MoExtend-Full     | -3.30（MoE LLM 全参数微调，遗忘显著）
MoE-LLaVA         | -7.86（MoE LLM 全参数微调，遗忘严重）
MoExtend          | -0.41（仅训练新增 expert，几乎无遗忘）
```

关键发现：MoE 的 expert 专业化使全参数微调时知识覆盖更严重——每个 expert 存储的特定领域知识被视觉相关梯度整体"冲刷"，不像 dense FFN 通过部分更新即可适应新模态。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **缓解策略分类**：
  1. **参数冻结 + 新增模块**（如 MoExtend）：原有参数不变，仅训练新增 expert/模块；对 MoE 尤其有效
  2. **正则化方法**（如 EWC, SI）：约束重要参数的更新幅度
  3. **数据回放**（Replay）：在微调数据中混入原有文本数据
  4. **参数高效微调**（如 LoRA, Adapter）：仅训练少量新增参数
- **MoE 特定的脆弱性根源**：MoE 的 expert 专业化 + sparse activation 导致：(a) 每个 expert 知识集中化，(b) 只有部分 expert 被 token 激活，(c) 被激活的 expert 通过梯度传播完整覆盖。相比之下 dense FFN 的分布式表示使知识覆盖更渐进
- **评估指标**（MoExtend 采用）：7 个纯文本 benchmark——ARC-Easy（常识推理）、HellaSwag（常识推理）、PIQA（物理常识）、Winogrande（常识推理）、MBPP（代码）、MMLU（聚合知识）、GSM8K（数学），使用 OpenCompass 评估框架，计算 Avg. drop

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension
