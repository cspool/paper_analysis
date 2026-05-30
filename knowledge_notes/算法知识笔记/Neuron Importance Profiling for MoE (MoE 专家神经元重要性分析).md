## Neuron Importance Profiling for MoE (MoE 专家神经元重要性分析)

术语解释
在 calibration samples 上对 MoE 每个 SwiGLU FFN neuron 进行重要性度量（四种方法），用于指导 neuron 按重要性排序重构为 major + minor sub-expert，支持 2T-Drop 的细粒度计算丢弃。

术语是什么？
四种 profiling 方法（在 calibration samples 上累积）：(1) Σ Swish(x·W₁^n)；(2) Σ |Swish(x·W₁^n)|；(3) Σ Swish(x·W₁^n) ⊙ (x·W₃^n)；(4) Σ |Swish(x·W₁^n) ⊙ (x·W₃^n)|。实验：(a) 绝对值方法优于非绝对值（避免正负抵消）；(b) 不同模型 affinity 不同（Mixtral+OLMoE：方法2最佳；DeepSeek：方法4最佳，因其含 shared expert 结构）；(c) 低负载 expert 出现大量负 gate value，高负载 expert 罕见。

从算法pipeline角度拆解术语：
```
For each expert e in MoE:
  importance = zeros(d_ffn)
  For each sample x in calibration (MMLU):
    importance += |Swish(x·W1_e) ⊙ (x·W3_e)|  # method 4 (best for DeepSeek)
  sorted_idx = argsort(importance, desc=True)
  major_idx = sorted_idx[:d_ffn/2]; minor_idx = sorted_idx[d_ffn/2:]
```

术语一般如何实现？如何使用？
- Calibration: MMLU 做 profiling（泛化性强），一次前向传播即可完成
- 与 expert partition 结合：先 partition (E→E×P)，再每 finer expert 做 profiling → 总 sub-expert = 2×E×P
- 局限：静态 profiling 无法捕捉 runtime dynamic patterns（但 gating score 跨任务稳定佐证其有效性）；profile method 需 per-model 选择；>2 split 可能进一步改善但降低 compute intensity

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
