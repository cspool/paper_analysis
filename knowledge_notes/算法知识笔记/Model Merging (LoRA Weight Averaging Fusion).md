## Model Merging (LoRA Weight Averaging Fusion)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model Merging (模型合并) 是 training-free 多任务能力集成技术：将不同下游任务上微调的 LoRA adapters 通过权重平均合并到同一 base model。标准方法 (Eq. 12)：W' = W₀ + (1/t)·Σ_{i=1}^t B_i·A_i。高级方法：TIES-MERGING (sign consensus + trimming)、DARE (dropout + rescale)。

核心痛点：不同任务 LoRA 参数 B_i·A_i 在参数空间中重叠——可训练 A_i, A_j 之间无正交性，合并造成 destructive interference (e.g., ScienceQA Δ=-60.34%)。FlyLoRA 通过冻结随机投影 A_i (近似正交) 解决。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Weight Averaging (FlyLoRA Eq. 12):
// 输入: task_i 的 (B_i, A_i), base W₀
W' = W₀ + (1/t) * Σ_{i=1}^t B_i·A_i

// FlyLoRA 优势: A_i, A_j 近似正交使:
// <B_i·A_i, B_j·A_j>_F ≈ 0 (Corollary 3.5)
// ||Σ w_i·B_i·A_i||²_F ≈ Σ w²_i·||B_i·A_i||²_F
//
// 合并性能降 Δ% (Llama-3.1-8B, weight averaging):
//               MMLU   ScienceQA  GSM8K   HumanEval
// LoRA(r=8):    -6.48  -60.34     -30.15  -13.04
// LoRA(r=32):   -4.91  -59.66     -31.48  -11.43
// Split-LoRA:   -4.86  -54.74     -28.30  -9.92
// FlyLoRA:      -2.02  -43.05     -21.81  -4.27
//
// CKA (Table 16): FlyLoRA 0.85/0.53/0.71/0.84 vs LoRA 0.78/0.39/0.58/0.75
// 更高 CKA → 合并后与单任务输出对齐更好
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- HuggingFace PEFT `add_weighted_adapter()`, mergekit (https://github.com/arcee-ai/mergekit)
- TIES-MERGING: trim (移除低幅值) → elect sign (多数决定) → disjoint merge
- DARE: delta params 随机 dropout p% → rescale 1/(1-p) → merge
- FlyLoRA 与 TIES/DARE 兼容 (plug-and-play), 叠加效果更佳 (Table 12)

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
