## Backward Stream Scheduling in MoE Training (MoE训练反向流调度 / All-to-All Prioritization)

术语是什么？
Backward Stream Scheduling 是 PopFetcher 提出的一种 MoE 训练 backward pass 中的 CUDA stream 优先级调度机制。由于 expert prefetching 导致同一 expert 可能存在多个副本在不同 worker，backward pass 中需要额外的 All-Reduce 操作来聚合 prefetched expert 的梯度。同时存在三种 CUDA stream：(1) EP All-to-All（token 回传）；(2) non-MoE All-Reduce（梯度聚合）；(3) prefetched expert All-Reduce（副本梯度回主 expert）。在 NCCL 中通信原语在 CUDA stream 被锁定在调用点，无法实时调整优先级。PopFetcher 将 All-to-All 和 All-Reduce 通信分解为 micro-operations 交替流水线执行，All-to-All 优先级高于 All-Reduce。

从kernel调度角度拆解术语：
Backward pass 的 micro-operation 流水线调度：
```
// 传统的 backward pass stream 争抢问题：
// Stream1: All-to-All (token 回传)
// Stream2: All-Reduce (non-MoE gradient)
// Stream3: All-Reduce (prefetched expert gradient)
// 问题：三个 stream 并发竞争 NCCL bandwidth → All-to-All 被非关键 All-Reduce 阻塞

// PopFetcher 的 micro-operation pipelining:
stream_priority = {A2A: HIGH, AR_NON_MOE: MEDIUM, AR_PREFETCH: LOW}

// 将 All-to-All 分解为 micro-ops
a2a_micro_ops = split_into_chunks(all_to_all_data, chunk_size)
// 将 All-Reduce 分解为 micro-ops  
ar_micro_ops = split_into_chunks(all_reduce_data, chunk_size)

// 交错执行，A2A 优先
while a2a_micro_ops or ar_micro_ops:
    if a2a_micro_ops:
        execute_next(a2a_micro_ops)     // 优先 All-to-All
    elif ar_micro_ops:
        execute_next(ar_micro_ops)      // All-to-All 完成后再 All-Reduce
```

效果：减少 backward computation blockage 10.9%（MoE-GPT）、10%（MoE-BERT）。核心原理是保证 token 回传不被 gradient 聚合阻塞——token 回传是下一层计算的依赖，而 gradient 聚合仅影响权重更新（可稍延迟）。

术语一般如何实现？如何使用？
在 C++ 和 CUDA 中实现 pipeline scheduling：将通信操作分解为微操作后，通过 CUDA event 和 stream 管理执行顺序。NCCL 层面无法直接修改优先级，因此通过在应用层控制 micro-operation 的提交顺序来实现效果。适用于 MoE 训练中同时存在 All-to-All 和 All-Reduce 的场景，尤其在采用了 expert prefetching/replication 后 prefetched expert 的额外 All-Reduce 会加剧 stream 竞争。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
