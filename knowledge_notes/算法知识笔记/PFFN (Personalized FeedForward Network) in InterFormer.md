## PFFN (Personalized FeedForward Network) in InterFormer

术语是什么？
PFFN（Personalized FeedForward Network）是InterFormer推荐模型架构中的个性化前馈网络组件，用于实现sequential features（浏览历史）和non-sequential features（用户人口统计）之间的bidirectional information flow。PFFN由五个顺序操作组成：(1) batched matrix multiplication with bias (FFN layer 1)；(2) GELU activation；(3) RMSNorm（root-mean-square normalization）；(4) FFN layer 2 (batched matrix multiplication with bias)；(5) 最终RMSNorm。处理tensor X ∈ R^{B×N×D}，权重矩阵W1 ∈ R^{B×D×K}和W2 ∈ R^{B×K×D}。

从算法pipeline角度拆解术语：
```
PFFN module forward pass:
  Input: X ∈ R^{B×N×D}, W1 ∈ R^{B×D×K}, W2 ∈ R^{B×K×D}
  
  Step 1: H1 = X @ W1 + b1              # FFN layer 1: batch matmul
  Step 2: H2 = GELU(H1)                  # Activation
  Step 3: H3 = RMSNorm(H2)               # Normalization
  Step 4: H4 = H3 @ W2 + b2              # FFN layer 2: batch matmul
  Step 5: Output = RMSNorm(H4)           # Final normalization
  
  Production shapes: (B, N, D, K) ∈ {
    (1024, 200, 256, 160), (1024, 200, 192, 96),
    (1024, 400, 256, 160), (1024, 150, 96, 192)
  }
```

PyTorch baseline使用torch.compile生成两个独立kernel：(1) extern_kernels.bmm（单pass：load→compute→write），(2) triton_per_fused_rms_norm_add_gelu（两pass：pass1 load+bias+RMSNorm statistics accumulation，pass2 reload+normalization应用）。总计3次memory round-trips。

KernelEvolve生成single-pass fused kernel：tile加载一次→完成全部5个operations（matmul+bias+GELU+RMSNorm+matmul+bias+RMSNorm）→写回HBM，仅需1次load+1次write per tile。Production shapes上实现1.2-2.6× speedup。

术语一般如何实现？如何使用？
PFFN fused kernel通过shape-specific tiling和cross-operation tile reuse实现优化。对于D∈[96,256]、K∈[96,256]的production shapes，tile尺寸适配SRAM容量以保证full operator chain on-chip execution。Batch size增大时speedup从2.0-2.6× (B≤256) 收敛到1.2-1.4× (B>512)，因为更大的batch amortizes了kernel launch overhead。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
