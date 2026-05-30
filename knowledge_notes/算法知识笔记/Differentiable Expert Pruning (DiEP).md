## Differentiable Expert Pruning (DiEP)

术语解释
DiEP（Differentiable Expert Pruning）是一种将 MoE 专家剪枝重新表述为连续优化问题的框架。通过定义可学习的 intra-layer importance scores（α）和 inter-layer importance scores（β），将原本指数级增长的离散专家搜索空间转化为可微的连续空间，利用梯度下降实现全局最优的非均匀专家剪枝。由 Bai et al. (2025) 在 NeurIPS 2025 提出，是首个将 differentiable architecture search 思想应用于 MoE 架构的方法。

术语是什么？
DiEP 的核心思想是将专家选择从离散的 binary mask m_i^(l) ∈ {0,1} 转化为连续的加权聚合：

y'^(l+1) = β^(l) · Σ_i ᾱ_i^(l) · FFN_i(x^(l))

其中 ᾱ_i^(l) = softmax(α_i^(l)) 为层内专家重要性（归一化后），β^(l) 为跨层重要性标量。目标函数：

L(α, β) = L_ce(y, F'(x; α, β)) + λ · ∥F'(x; α, β) − F(x)∥_F

包含两部分：(1) Cross-entropy loss 保持任务性能；(2) Reconstruction Regularization（Frobenius norm）鼓励剪枝后模型输出与完整模型一致。λ 为平衡系数（Mixtral 上 λ=0.01）。

优化采用 Alternating Update Strategy：以 α:β = 3:1 的比例交替更新，解耦两个参数组的梯度路径，避免 DiffPruning 等先前方法中的 gradient conflict 问题。

收敛后，全局重要性 s_i^(l) = α_i^(l) · β^(l)，对所有 L×N 个专家统一排序，按 sparsity ratio r 删除底部 K = N·L·r 个最不重要专家。这实现了自动的非均匀剪枝：浅层（1-15 层）因 β 和 α 值更高自然保留更多专家，深层冗余大的层剪去更多专家。

从算法pipeline角度拆解术语：
```
# DiEP Algorithm Pipeline
Input: Full MoE model F, calibration data D_cal (128 samples)
       α_i^(l) = 1 ∀i,l,  β^(l) = 1 ∀l,  λ = 0.01

# Phase 1: Differentiable Search (10 epochs)
for epoch in 1..10:
  for batch in D_cal (batch_size=16):
    # Forward with continuous relaxation
    for layer l in 1..L:
      ᾱ_i^(l) = softmax(α_i^(l))              # ∈ R^N
      for expert i in 1..N:
        h_i = FFN_i(x^(l))                      # expert forward
      y'^(l+1) = β^(l) · Σ_i ᾱ_i^(l) · h_i    # weighted aggregation (Eq.5)
    
    F'(x) = full forward with weighted experts
    L = L_ce(y, F'(x)) + λ · ∥F'(x) − F(x)∥_F  # Eq.7
    
    # Alternating updates (3:1 ratio)
    for step in 1..3:                             # α updates
      α ← α − η_α · ∇_α L(α, β)                 # η_α = 5e-3, cosine schedule
    for step in 1..1:                             # β updates  
      β ← β − η_β · ∇_β L(α, β)                 # η_β = 5e-3

# Phase 2: Global Pruning
K = N_layers × N_experts × r                      # e.g., 32×8×0.5=128 experts to prune
for each expert (l, i):
  s_i^(l) = α_i^(l) · β^(l)                       # global importance (Eq.10)
P = argsort(s)[:K]                                # bottom-K least important
for (l, i) in P:
  remove expert i from layer l                    # permanent pruning

# Phase 3: Optional Merging
for each pruned expert e_p:
  e_retained = argmax CKA(e_p, e_j) over retained experts
  merge e_p into e_retained with CKA-based weight
```
关键维度：α ∈ R^(L×N)，β ∈ R^L。额外参数量仅 ~0.01%。Mixtral 8×7B pruning time: 0.23h（vs NAEE exhaustive search 1.31h）。Deepseek-MoE-16B (64 experts/layer) pruning: 0.28h（vs NAEE ≈94000 days 因搜索空间爆炸不可行）。

术语一般如何实现？如何使用？
- 实现依赖：HuggingFace Transformers + lm-eval-harness。Calibration 仅需 128 C4 samples
- 超参数：epochs=10, batch_size=16, lr=5e-3 (cosine schedule), λ=0.01 (Mixtral), α:β update ratio=3:1
- 剪枝后模型标准 HuggingFace 格式，可直接加载推理
- 论文未提供开源代码，但方法可基于标准 PyTorch 复现（核心仅 ~300 行参数更新逻辑）
- 支持 optional expert merging（CKA-based）进一步提升性能
- 论文未在 DeepSeek-V3/Qwen2.5-Max 等更大模型上验证（计算资源限制）

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

---
