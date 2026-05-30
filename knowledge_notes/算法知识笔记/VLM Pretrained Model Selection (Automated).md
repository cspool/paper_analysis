## VLM Pretrained Model Selection (Automated)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLM Pretrained Model Selection 是 Mordal 首次定义的问题：给定下游任务、alignment 数据集和 pretrained 模型库（VE 集合 + LLM 集合），在资源约束下找到最优 VE×LLM 组合，使得经过 vision-text alignment 训练后在目标任务性能最优。核心难点：未 alignment 的 VLM 无法评估 zero-shot 性能（feature projector 未训练时 LLM 不理解 image embeddings，会产生随机输出），因此传统 model selection 方法（LogME, LEEP 等，设计用于 vision-only 或 text-only 任务）在此场景失效——必须实际训练 projector 后才能评估。Mordal 将其建模为资源约束的组合优化问题，minimize search cost while maximizing selection accuracy。

从算法pipeline角度拆解术语，给出具体例子。
问题形式化：
```
Given: M_ve × M_llm (m×n candidates), D_align, D_task, Budget B
Goal: (VE*, LLM*) = argmax Perf(align(VE, LLM, D_align), D_task)
      s.t. total search cost ≤ B

Naive grid search: m=7, n=7 → 49 candidates
  Each: ~111 GPU hours (projector training + LoRA fine-tuning on A40)
  Total: 5439 GPU hours on 16×A40
  All found best VLMs surpass LLaVA-1.5-7B equivalent (CLIP-Vicuna)

Challenge: chicken-and-egg — must train to evaluate, cannot pre-filter
```

Mordal 三阶段 solution：(1) CKA clustering 将相似候选分组减少搜索空间；(2) SHA early stopping 快速淘汰差 clusters；(3) Scaling prediction 从部分数据训练预测完整性能。

术语一般如何实现？如何使用？
开源：https://github.com/SymbioticLab/Mordal。接口：`mordal.query_for_model(data, task, pretrained_ve_zoo, pretrained_llm_zoo, ...)`。用户提供 alignment data + target task data + model zoo list。支持配置 projector 架构（MLP/Linear）、freeze 策略（freeze VE, LoRA fine-tune LLM）、clustering/exploration/prediction 超参。在 7VE×7LLM×6 tasks 评估中 8.9×–11.6× speedup，5/6 任务选出 Top-1。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---
