## Ring Attention (Sequence-Parallel Distributed Attention)

术语是什么？
Ring Attention是一种序列并行(Sequence Parallelism)方法，将长序列的KV tensors按块分片到多个GPU上。每个GPU对其本地KV块执行blockwise attention计算，同时将KV块沿GPU环(ring)传输到下一个GPU。计算和通信重叠：GPU计算当前KV块的attention时，同时将前一块的KV异步发送给peer。这样每个token最终能attend到序列中的所有token，而不需要将整个KV cache复制到每个GPU。

从算法pipeline角度拆解术语：
Ring Attention核心算法流程（以8 GPU, 序列长度S为例，每GPU持有S/8个token的Q分片）：
```
for step in range(num_gpus):
    kv_rank = (local_rank + step) % num_gpus
    // 1. 计算当前KV块的blockwise attention
    attn_score = Q_local @ K[kv_rank].T  // scaled dot-product, 本地softmax累加
    attn_output += p @ V[kv_rank]        // softmax_weighted V累加
    // 2. 异步发送/接收KV块 (与step+1重叠)
    if step < num_gpus - 1:
        async_send(K_local, V_local, next_gpu)
        async_recv(K_next, V_next, prev_gpu)
```
通信复杂度O(S)而非O(S²)。ParallelKittens通过fused单kernel实现inter-SM overlapping：专用communication SM将下一块KV批量传输到local HBM（避免remote L2 cache miss导致的重复传输），compute SM执行FlashAttention计算，SM分配自动调优。

术语一般如何实现？如何使用？
xDiT baseline使用NCCL P2P send + FlashAttention-3在独立CUDA stream上做coarse-grained overlap。ParallelKittens将其融合为单个kernel（LCSC模板），通过num_comm_sms控制SM分配，实现1.07x-4.08x speedup（B=16, H=16, D=128）。

涉及论文标题：
- ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels

---
