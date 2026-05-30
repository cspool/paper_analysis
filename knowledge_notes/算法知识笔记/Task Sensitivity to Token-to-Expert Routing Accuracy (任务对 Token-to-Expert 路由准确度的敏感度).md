## Task Sensitivity to Token-to-Expert Routing Accuracy (任务对 Token-to-Expert 路由准确度的敏感度)

术语解释
Task Sensitivity to Token-to-Expert Routing Accuracy 是指不同 NLP 任务对 MoE 模型中 expert routing 错误的容忍度存在显著差异的观察。分类和语义相似度任务即使使用随机 routing 仍能保持高输出质量，而对话和摘要等开放式任务对 routing accuracy 高度敏感。这一发现是 task-aware expert loading 等系统优化的算法基础。

术语是什么？
MoE 模型的 router gate 决定每个 token 由哪些 expert 处理。当部分 MoE layer 的 routing 不准确（token 被路由到非最优 expert）时，不同任务的输出质量退化程度不同。eMoE 通过 progressive inaccurate routing 实验（从靠近 input 的层开始逐层应用 random routing，测量输出与 full-model ground truth 的 BERT semantic similarity）：

**实验结果（eMoE §2.2.3，Figure 5）**：
- **Classification / Comparison tasks**：即使 100% layers routing 不准确，similarity 仍 >90%
- **Conversation / Summarization tasks**：75% layers 准确时 similarity 已 <80%
- **QA tasks**：50% layers 准确时 similarity 仍在 >80%
- **Summarization tasks**：50% layers 准确时 similarity <80%

根因分析：靠近 input 的神经网络层倾向于学习 general representations，靠近 output 的层 specialize 为 task-specific representations。任务对 input-side layers 的 routing accuracy 的依赖度不同——开放式生成任务需要更精确的语义分解，分类任务仅依赖高层语义特征。

从算法pipeline角度拆解术语：
MoE 推理中 token routing 与 task sensitivity 的关系：

```
=== MoE Layer 的 Token-to-Expert Routing ===

Input: token embedding x ∈ R^d
For each MoE layer L_i (i = 1..m):
  # Step 1: Gate computation
  g = softmax(W_g · x)  # W_g ∈ R^{d × N_experts}
  
  # Step 2: Top-K selection (accurate routing)
  top_k_indices = argtopK(g, k)  # e.g., k=2
  top_k_weights = g[top_k_indices]
  
  # Step 3: Expert FFN computation
  output = Σ_{j∈top_k} top_k_weights[j] · Expert_j(x)
  # Expert_j(x) = W_out_j · σ(W_in_j · x)

=== Inaccurate Routing (simulated in sensitivity experiment) ===
For first L inaccurate layers (L = 0%..100% of total MoE layers):
  # Replace accurate routing with random selection
  random_k_indices = random_sample(N_experts, k)
  random_k_weights = uniform(k)  # equal weights
  output = Σ_{j∈random} random_k_weights[j] · Expert_j(x)

For remaining (m-L) accurate layers:
  output = standard top-K routing  # as above

=== Task Sensitivity Metric ===
sensitivity(task, layer_range) = 
  BERT_similarity(output_with_inaccurate_routing[0:layer_range], 
                  output_with_full_accurate_routing)

# Classification: s ≈ 0.9+ even when layer_range = 100%
# Conversation: s < 0.8 when layer_range = 75%
```

术语一般如何实现？如何使用？
- Sensitivity profiling 是 offline 一次性操作：对每个 task type 在目标 MoE model 上运行 progressive inaccurate routing 实验
- 结果存储为 per-task per-layer sensitivity matrix：`s[T][L] ∈ {0, 1}`（1=sensitive, 0=insensitive），threshold 通常设为 85% similarity
- 应用：Task-aware Expert Loading（eMoE）、task-aware model compression、task-specific expert pruning
- 与 Layer-wise Importance 的区别：layer importance（如 early exit 文献）通常仅考虑 layer depth，不考虑 task type 维度
- 与 existing work 的联系：Zeiler & Fergus (ECCV 2014) 发现 lower layers learn general features, higher layers learn task-specific features —— 这一神经网络原理在 MoE routing 语境下被 eMoE 扩展到 task type 维度

涉及论文标题：
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

---
