## Factorization Machine (FM) and Optimized FM

术语是什么？
Factorization Machine (FM) 是推荐系统中的特征交互建模方法，通过分解的隐向量（latent vectors）捕获sparse categorical features之间的pairwise interactions。标准FM计算XX^T（O(N²D) complexity）。Wukong的Optimized FM引入learnable projection matrix Y ∈ R^{N×K} (K << N)，利用associativity将计算重组为out = X · (X^T Y)，将复杂度从O(N²D)降低至O(NKD)。这是通过low-rank approximation实现的——用N×K投影替代N×N全pairwise matrix。

从算法pipeline角度拆解术语：
```
Standard FM (pairwise dot product):
  out_{ij} = <x_i, x_j> for all feature pairs (i,j)
  Complexity: O(N²D) — prohibitive for thousands of features

Optimized FM (Wukong):
  Step 1: X^T Y = torch.bmm(x.permute(0,2,1), y)  # (B,D,N) @ (B,N,K) = (B,D,K)
  Step 2: out = torch.bmm(x, xty)                   # (B,N,D) @ (B,D,K) = (B,N,K)
  Complexity: O(NKD) — reduced from O(N²D) since K << N
  
  Production shapes (WuKong variant): (B, N, D, K) ∈ {
    (1024, 24, 224, 2198), (1024, 40, 224, 448), (1024, 48, 224, 448)
  }
```

KernelEvolve将两步bmm融合为单个Triton kernel：X^T Y intermediate result保持在SRAM中，消除HBM round-trip（从2次load+2次write减少到1次load+1次write），在production shapes上实现2-4× speedup。

术语一般如何实现？如何使用？
Optimal FM是WuKong recommendation model的核心primitive。KernelEvolve生成的fused kernel通过shape-specific tiling（tile尺寸适配SRAM容量以保证full computation chain on-chip）和cross-operation tile reuse（同一tile的load完成两次matmul后才写回HBM）实现优化。当feature count N增大到tiling overhead超过fusion benefit时（N > 64），系统自动fallback到PyTorch unfused baseline。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
