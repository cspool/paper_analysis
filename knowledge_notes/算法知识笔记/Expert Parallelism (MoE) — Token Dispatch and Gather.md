## Expert Parallelism (MoE) — Token Dispatch and Gather

术语是什么？
Expert Parallelism将MoE层的多个expert FFN权重均匀分布到不同GPU，输入tokens通过router选出top-k expert后，经all-to-all dispatch发送到对应GPU，执行expert MLP计算，再all-to-all gather回原GPU。通信开销源于两次all-to-all，消息粒度细（per-token KB级），对通信库的连续大块传输设计不友好。

从算法pipeline角度拆解术语：
```
// 输入tokens: (T, H), 每GPU T/G个token, E experts, k=top-k
// Step 1: Router (local per GPU)
logits = tokens @ W_router            // (T/G, E)
topk_vals, topk_idx = topk(logits, k) // per-token top-k expert assignment

// Step 2: All-to-All Dispatch (token → expert GPU)
permuted_tokens = all_to_all(tokens, indices=topk_idx)  // 每个token发送到持有其topk expert的GPU
// 通信量: T × d_model bytes

// Step 3: Expert MLP Compute (per GPU, independent)
for each received_token:
    expert_output = FFN_expert(received_token, W_expert)

// Step 4: All-to-All Gather (result → original GPU)
output = all_to_all(expert_output, reverse_indices)
```
瓶颈：fine-grained all-to-all通信，每token需发送到up to k个expert GPU。PK通过TMA tile级P2P与Grouped GEMM做intra-SM overlap，fused kernel <40行device code，vs Comet手写kernel达成0.92-1.22x性能。

术语一般如何实现？如何使用？
通信库：DeepEP (DeepSeek)、Comet、FlashDMoE。PK: TMA store_async + intra-SM overlap。与Data/Tensor Parallelism混合使用（3D parallelism）。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
