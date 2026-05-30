## Rank-Wise Expert Allocation (in MoE-based LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlyLoRA 提出的极限 expert 分解策略：将 LoRA 的 r 个 rank 每个作为独立 expert——共有 r 个 rank-1 expert，每 token 仅 top-k 个被激活 (k<r)。f(x) = W₀·x + (α/r)·Σ_{i∈I_topk} b_i·(a_i·x)，其中 a_i=A[i,:]∈R^{1×n}, b_i=B[:,i]∈R^{m×1}。

理论依据 (Theorem 3.3)：top-k 稀疏性使 off-diagonal 梯度协方差按 k²/r² 缩减。k=8, r=32 时 off-diagonal 协方差约为 dense 的 1/16；k=1 时几乎完全去耦合。图 3(b-c) 梯度相关热力图验证：LoRA-FA(r=32) 密集相关，FlyLoRA(k=8) 显著稀疏。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Rank-Wise Expert Allocation:
// 将 B 矩阵按列分解为 r 个 rank-1 "expert"
// 每个 expert_i = b_i · a_i · x

Forward:
  y = A @ x                             // [r]
  I_topk = argtopk(y + d, k)            // k 个 rank-1 expert 被激活
  mask = zeros(r); mask[I_topk] = 1
  delta = (α/r) * (B ⊙ mask) @ y       // 仅 k 列计算

// 梯度去耦合 (Theorem 3.3):
// E[g_i^sparse · g_j^sparse] ≈ (k²/r²) · E[g_i^dense · g_j^dense]
//
// 消融 (图 4b): 固定 r=32, 变 k:
// k 太小 → 信息不足; k 太大 → interference 增加
// 最优 k=8~12; 默认 k=8
//
// 消融 (图 4c): 固定 k=8, 增加 r:
// 性能持续改善 (更多 capacity 无额外 interference)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch 实现：A ∈ R^{r×n} 作为 frozen nn.Linear weight, B ∈ R^{m×r} 作为 trainable nn.Linear weight
- Top-k selection：torch.topk(y, k).indices, boolean mask 乘 B forward
- Backward：grad 通过 mask 自动归零 (PyTorch autograd)

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
