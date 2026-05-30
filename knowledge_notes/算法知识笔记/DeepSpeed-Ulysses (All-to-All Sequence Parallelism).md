## DeepSpeed-Ulysses (All-to-All Sequence Parallelism)

术语是什么？
DeepSpeed-Ulysses在self-attention前后执行all-to-all通信：进入attention前从sequence-sharded切换到head-sharded（每GPU拥有所有token但仅部分head），使self-attention可直接head-independently执行而无需通信；attention后all-to-all恢复sequence-sharded。与Ring Attention的渐进式块传输不同，Ulysses通过两次全量4D all-to-all完成并行，通信开销集中在all-to-all阶段。

从算法pipeline角度拆解术语：
```
// 8 GPU, 输入X: (B, S/8, H, D) per GPU — sequence-sharded
// Step 1: S→H shard转换 (all-to-all)
X_head = all_to_all(X, scatter_dim=S, gather_dim=H)  // (B, S, H/8, D)
// Step 2: head-independent self-attention (无通信)
Y = attention(X_head)  // per-GPU Q/K/V shapes: (B, S, H/8, D)
// Step 3: H→S shard恢复 (all-to-all)
Y_seq = all_to_all(Y, scatter_dim=H, gather_dim=S)     // (B, S/8, H, D)
```
瓶颈：all-to-all沿inner dimension（H/S dim），NCCL不支持非连续layout的直接collective，baseline需先reshape再通信。PK实现细粒度4D tile级P2P all-to-all绕过此overhead，kernel代码<50行。

术语一般如何实现？如何使用？
YunChang baseline: NCCL+reshape。ParallelKittens: Tile级P2P直接传输，4D (B,S,H,D) all-to-all。达成1.01x-1.39x speedup（B=16, H=128, D=128）。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
