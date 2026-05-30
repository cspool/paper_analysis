## WY Representation (for Householder Products)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WY 表示由 Bischof & Van Loan (1985) 提出，将 Householder 反射矩阵乘积 Π_i (I - β_i v_i v_i^T) 紧凑表示为 I - W Y^T。在 DeltaNet/Gated DeltaNet 中，每步 transition 为 Householder 形式 (I - β_t k_t k_t^T)，其累积乘积 P^r = Π_{i=1}^r (I - β_i k_i k_i^T) 可表示为 I - W K^T，其中 W = T K，T = (I + strictLower(diag(β) K K^T))^{-1} diag(β)。这使得 O(L·d²) 串行过程转为 O(C·d²) matmul 并行计算，是利用 Tensor Core 的关键。Gated DeltaNet 的扩展：在 T 矩阵计算中加入 Γ ⊙ K K^T 以融入 gating。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 原始：串行 Householder 链
P^r = Π_{i=1}^r (I - β_i k_i k_i^T)

// WY 表示后：
P = I - W K^T  // 其中 W = T @ K
T = solve_triangular(I + strictLower(diag(β) K K^T), diag(β))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
核心实现为 BLAS Level 3 的 triangular solve (trsm)。PyTorch 中用 torch.linalg.solve_triangular。使 DeltaNet/GatedDeltaNet 训练从不可行变为仅比 Mamba2 慢约 10%。Joffrain et al. (2006) 的 UT transform 进一步优化了表示计算。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---
