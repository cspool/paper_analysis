## Expert Prediction for MoE Inference (MoE 推理中的 Expert 预测)

术语解释
Expert Prediction for MoE Inference 是一种使用机器学习模型（特别是序列预测模型）在推理前预测即将需要的 expert 集合，从而仅将预测的 expert 加载到 GPU 显存中的技术。与基于当前层 gate 输出的 per-layer prefetching 不同，expert prediction 在 prompt 处理前一次性预测全部 MoE 层的 expert 序列，属于 proactive prediction 而非 reactive prefetching。

术语是什么？
eMoE 提出的 Expert Prediction 将 MoE expert 选择建模为序列预测任务：给定过去 prompt/层的 expert 路由历史，预测未来 prompt/层的 expert 分布。核心要素：

1. **Expert 序列表示**：MoE 模型有 m 层，每层选 top-k expert。Expert 序列表示为 `[e_1, e_2, ..., e_m]`，其中 `e_i = [k 个 expert index]`。
2. **预测模式**：
   - **eMoE-A（All-layer）**：`f([e_1^{r1}, ..., e_m^{r1}]) → [e_1^{r2}, ..., e_m^{r2}]`，用前一条 prompt 的完整 expert 序列预测当前 prompt 的全部 expert。
   - **eMoE-L（Layer-by-layer）**：`f(e_{i-1}^{r1}) → e_i^{r1}`，用上一层的 expert 预测当前层的 expert。
3. **预测器选择**：BERT-XLNet（0.108B params），优于 GPT-2（50%-51% accuracy）和 BERT-base（13%-16%）。XLNet 通过 permutation language modeling 学习双向上下文，捕获序列元素间更复杂的关系。最终 accuracy：~71%（Mixtral）、~70%（OpenMoE）。
4. **Memory 开销**：predictor 仅占 MoE model 的 0.24%-1.3% memory。
5. **Fallback**：预测错误的 expert 未在 GPU 上时，token 被路由到已加载的 next top-k expert。

预测的基础观察（eMoE §2.2.1, §3.2.2）：
- Consecutive layer 间 expert 选择有 ~0.50 cross-correlation（gate network 使用类似 basic features）
- Consecutive prompt 间 correlation 0.75-0.95（Mixtral），因 shared context/vocabulary

从系统架构角度拆解术语：
Expert Prediction 在 eMoE 推理系统中的工作流程：

```
=== Expert Prediction Pipeline ===

Input: request stream (prompts), expert routing history

Phase 1 — Data Collection:
  For each processed prompt:
    Record: router gate top-k expert selection per layer
    Store: expert_index_sequence = [(l1: e_a, e_b), (l2: e_c, e_d), ..., (lm: e_x, e_y)]

Phase 2 — Training (offline):
  Training data: 80% expert routing traces from target MoE model on task dataset
  Test data: 20%
  Labels: expert indices (numerical digits)
  Model: BERT-XLNet (HuggingFace pretrained, 0.108B params)
  Task: sequence prediction — predict next element(s) given past elements

Phase 3 — Online Prediction (每 p prompts 调用一次):
  eMoE-A:
    Input: previous prompt's expert sequence [e_1^{prev}, ..., e_m^{prev}]
    Forward: XLNet.forward(input_sequence)
    Output: predicted expert sequence [e_1^{curr}, ..., e_m^{curr}]
    Each e_i = distribution over experts → argmax → top-k

  eMoE-L:
    For layer i in 1..m:
      Input: expert indices from layer i-1
      Forward: XLNet.forward(layer_{i-1}_experts)
      Output: predicted experts for layer i

Phase 4 — Expert Loading Decision:
  Compare: predicted experts vs currently loaded experts on GPU
  New = predicted ∖ currently_loaded   → copy to GPU (async)
  Evict = currently_loaded ∖ predicted → move to CPU
  Budget: pick top L experts per layer (L set by memory budget),
          sorted by expected token count N_i = (ΣW_j + T·W_o) · s · f_i

Phase 5 — Fallback at Inference:
  Router gate: top-k expert selection
  For each selected expert:
    if expert in GPU: use it
    else: use next top-k expert on GPU
```

术语一般如何实现？如何使用？
- 实现框架：PyTorch（预测器模型），HuggingFace Transformers（XLNet pretrained weights）
- 运行位置：GPU 上的独立进程（与 inference engine 并行）
- 预测器选择标准：accuracy vs latency trade-off。eMoE-A 时间开销 ~0.334s-0.381s per call，eMoE-L ~1.387s-4.211s per call
- Interaction with inference engine：Python multiprocessing lock per MoE layer + CUDA event 同步 expert 加载与计算
- 适用场景：长 running MoE inference serving with stable task distribution
- 关键限制：预测 accuracy ~70-71%，fallback routing 可能使 token 被路由到次优 expert

涉及论文标题：
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference
