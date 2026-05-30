## Gradient-Free Post-Training Compression (无梯度后训练压缩)

术语是什么？
无梯度后训练压缩是一类不依赖反向传播和梯度计算的模型压缩方法的统称，其核心思想是在模型训练完成后，仅使用推理能力对模型进行压缩。与传统压缩范式（Prune → Fine-tune with SGD）不同，无梯度方法不更新或仅通过 weight averaging/merging 方式更新模型参数。

EEP 将无梯度压缩推进为一种完整的 paradigm：(1) 通过进化搜索实现 expert 选择（pruning），(2) 通过 continuous weight merging 实现知识恢复（analogous to fine-tuning but without gradients）。整个过程不涉及任何 Pytorch backward() 调用。
- 传统范式 I：Importance-based selection + SGD fine-tuning（需要 GPU ≥ 2× model size）
- 传统范式 II：Selection + distillation（需要 teacher model 做额外推理）
- EEP 范式 III：Evolutionary selection + Weight merging（仅需推理设备，单次评估仅做 forward pass）

从算法pipeline角度拆解术语。
```
# 传统有梯度压缩 vs 无梯度压缩（EEP）
# 传统: Prune + Fine-tune
W_pruned = prune_by_importance(W, sparsity)    # e.g., magnitude pruning
for epoch in range(fine_tune_epochs):
    loss = CrossEntropy(model(W_pruned, x), y)
    loss.backward()                              # 需要 ≥2× model size GPU memory
    optimizer.step()

# EEP: Search + Merge (gradient-free)
for iter in range(search_iterations):
    W_candidate = Mutate(Crossover(select(P)))  # ES operations
    accuracy = model.forward(W_candidate, x, y) # inference only, no backward()
    P.append((W_candidate, accuracy))
W_final = best_individual(P)
W_merged = continuous_weighted_average(W_final)  # merging phase
```

为什么需要？/解决什么痛点？
- **降低压缩门槛**：传统 fine-tuning 需要大显存 GPU（如 Mixtral 8×7B FP16 约 94GB，SGD 需要 >188GB），EEP 仅需推理显存（~44GB for fp16 或通过量化进一步降低）
- **设备灵活性**：可在推理专用硬件（如推理卡、边缘设备）上执行压缩
- **避免灾难性遗忘**：不使用梯度更新，通过 weight averaging 保留原始训练知识
- **适用于 downstream tasks**：EEP 针对特定下游任务数据集搜索，同时保持 zero-shot 泛化能力（MMLU OOD 实验验证）

术语一般如何实现？如何使用？
- EEP 实现：在 HuggingFace 框架上仅使用 model.generate() 和 accuracy computation，不调用 backward()。搜索完成后导出为标准 HF 格式模型权重。
- 搜索过程 40+160=200 iterations，每个个体一次推理（约数秒到数分钟），总搜索时间取决于种群规模和并行度
- 适用范围：适合无法负担 fine-tuning 计算资源的用户/场景，尤其适合针对特定下游任务数据集定制的模型压缩
- 限制：搜索成本仍不可忽视（论文标注为 limitation），无梯度搜索可能不如梯度-based fine-tuning 在极低 sparsity 下精确
- 代码：https://github.com/imagination-research/EEP

涉及论文标题：
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs
