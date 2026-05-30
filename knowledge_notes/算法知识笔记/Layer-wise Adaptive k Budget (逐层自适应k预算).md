## Layer-wise Adaptive k Budget (逐层自适应k预算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-wise Adaptive k Budget 是一种在 fixed total compute budget（总 k 值固定为 Σk_ℓ）条件下，通过非均匀分配每层 attention 的 k 值来最大化模型性能的技术。核心观察：(1) 不同层的 attention 稀疏度不同——Layer 1 entropy 最高（attention 最分散），后续层 entropy 迅速下降；(2) 后续层的 attention 更集中、更 sparse，意味着可以用更小的 k 而不损失性能；(3) 在 fixed total budget 约束下，将 k budget 从前层向后层线性递增（而非 uniform 分配），可以在不增加总计算量的前提下获得 non-trivial performance boost。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Uniform vs Adaptive k Budget 对比**：
```
# Total budget: K_total = Σ k_ℓ (固定，如 total k per layer sum = 32*128 = 4096)

# Strategy 1: Uniform
for ℓ in 1..L:
    k_ℓ = K_total / L      # 每层相同，如 128

# Strategy 2: Linear Increasing (Exploiting Sparsity)
for ℓ in 1..L:
    k_ℓ = k_base + (ℓ/L) * k_slope  # 第一层最小，最后一层最大
    # k_base + k_slope/2 = K_total/L (保持总 budget 不变)

# Decoding 时使用对应的 k_ℓ
for ℓ in 1..L:
    vals, I = K_index[ℓ].search(q.cpu(), k_ℓ)  # 逐层不同的 k
    attn_out = softmax(vals/sqrt(d_k)) @ V_ℓ[I]
```

术语一般如何实现？如何使用？

实现思路：(1) Offline profiling——在不同 task 上分析每层的 attention entropy 分布，确定各层相对 k 需求；(2) 基于 entropy 的自动分配——Pearson correlation 0.847 表明 entropy 是预测 k 需求的有效 proxy，可按 entropy 比例分配 k；(3) 使用方式简单——仅需在 Faiss search 阶段传入不同的 k 参数，无需修改其他 pipeline 组件。

Exploiting Sparsity 论文（Fig.9）在 RULER benchmark 上的实验显示：在相同 total k budget 下，linear increasing 策略优于 uniform 策略，对于某些 k budget 可获得 ~2% 的 RULER score 提升。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---
