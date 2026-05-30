## Non-MoE Computation Overlap (非MoE计算重叠)

术语是什么？
在 MoE 训练中，Non-MoE Computation Overlap 指利用 Transformer block 中非 MoE 组件（如 Attention 层、LayerNorm）的执行时间窗口，将通信或辅助计算与之重叠执行，从而隐藏延迟。PopFetcher 利用 Attention 层计算期间 network link idle 的特点，在此期间异步预取下一 MoE layer 的热门 expert 参数。Attention 层仅使用本地数据（无跨机通信），因此其执行期间 100% 的 network bandwidth 可用于 expert prefetching。

从kernel调度角度拆解术语：
Overlap 的 timeline 调度：
```
// 一个 Transformer block 的执行 timeline（单个 GPU worker）：
Time → 
[Attention Forward]  [MoE Forward: A2A Dispatch → Expert FFN → A2A Combine]  [Attention Backward]  [MoE Backward]
|<-- Non-MoE -->|    |<------------------- MoE ------------------->|       |<-- Non-MoE -->|    |<--- MoE --->|

// PopFetcher 的 overlap 策略：
// 在 Non-MoE 期间的 idle network link 上：
[Attention Forward + Expert Prefetch(l+1)]  [MoE Forward(l) with prefetched experts(l)]
[Attention Backward + Expert Prefetch(l+1)] [MoE Backward(l) with stream pipelining]
```

Overlap 条件：Time^{non-MoE} ≥ Σ expert_prefetch_time，即 Attention 计算时间必须覆盖所有需要预取的 expert 参数的总传输时间。当 bandwidth 有限（compute-to-bandwidth ratio ε 高）时，Attention 计算时间相对充裕，overlap 最有效。

术语一般如何实现？如何使用？
基于 PyTorch CUDA stream 管理：主训练 stream 执行 Attention forward/backward，独立 prefetch stream（torch.cuda.Stream）执行 P2P expert 参数传输。通过 CUDA event 同步确保预取在下一 MoE layer 开始前完成。适用于所有 MoE 训练框架，只要 EP 下 Attention 层计算期间 network link idle。在 Cluster B (8×A10, 32Gbps) 的 bandwidth-constrained 环境下收益尤为显著（加速比 1.18-18.3×）。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
