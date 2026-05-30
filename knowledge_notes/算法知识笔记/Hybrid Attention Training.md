## Hybrid Attention Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid Attention Training 是利用 MoBA 与 full attention 参数等价性，在训练中在稀疏（MoBA）和稠密（Full）attention 模式间切换的策略。核心优势：MoBA 提供训练效率，full attention 恢复质量，两者共享同一套模型参数无需转换。

从算法pipeline角度拆解术语：
两种子策略：

**1. Two-stage MoBA/Full Hybrid**：
```
Stage 1: train(MoBA, 90% tokens)  → 高效长上下文训练
Stage 2: train(Full, 10% tokens)  → 恢复完整 attention 能力
结果：position-wise LM loss ≈ pure Full Attention
无 loss spike during switching
```

**2. Layer-wise Hybrid**：
```
前 L-N 层：MoBA（稀疏）
后 N 层：  Full Attention（稠密）
```
动机：SFT 中 prompt tokens 的 loss 被 mask，导致 MoBA 的稀疏梯度无法有效 backprop。最后几层 full attention 提供 dense gradient path。实验（Figure 5b）显示 SFT loss 随 full attention 层数增加而单调下降。

术语一般如何实现？如何使用？

训练脚本中通过 schedule 控制 attention_mode 切换。推理时 prefill 用 MoBA（加速），decoding 用 full attention（保证质量）。典型配置：Llama-8B 从 128K→1M continual pre-training，block_size=4096, top-k=12, 最后 3 层 full attention。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs
