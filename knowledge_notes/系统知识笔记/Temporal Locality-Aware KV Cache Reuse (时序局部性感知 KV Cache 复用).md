## Temporal Locality-Aware KV Cache Reuse (时序局部性感知 KV Cache 复用)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Temporal Locality of KV Cache 是 ShadowKV 发现并利用的解码阶段 KV cache 访问模式：相邻 decoding step 的 attention 分布高度相似，导致两个连续 step 选中的 top-k chunk 有 >60% 的重叠（hit rate）。基于这一观察，ShadowKV 维护一个 chunk index 环形缓冲区记录上一步的选中 chunk，解码时通过 index scan 跳过已缓存 chunk 的 key 重建和 value 取回。

从系统架构角度拆解术语：

```
// Temporal Locality Cache 机制
prev_indices: RingBuffer[size=k]  // 上一步选中的 chunk indices

// 每个 decoding step
I_new = ArgTopK(S_agg, k)         // 当前步选中的 chunk indices
I_hit = intersect(I_new, prev_indices)  // 命中的 chunk（无需重建）
I_miss = setdiff(I_new, prev_indices)   // 未命中的 chunk（需重建+取回）

// 仅对 I_miss 执行:
//   - Stream 1: K_miss = Gather(A, I_miss) @ B
//   - Stream 2: V_miss = cudaMemcpy(V_CPU[I_miss])
// I_hit 的 KV 对已在 GPU cache 中，直接使用

prev_indices = I_new  // 更新环形缓冲区
```

在 ShadowKV 配置下（chunk size=8, k=256, ~60% hit rate），该机制减少约 60% 的 key 重建 GEMM 和 value PCIe 取回，将等效带宽从 7.2 TB/s 理论值中减去 (1-α) 的 cache miss 开销后仍保持 3.6× A100 原生带宽。

术语一般如何实现？如何使用？

实现为 CUDA kernel 中的 index scan 操作，维护 fixed-size ring buffer。Temporal locality 的强弱取决于 chunk size 和 sparse budget：chunk 越小、budget 越大，相邻步 overlap 通常越高，但 chunk 过小会降低压缩效果。ShadowKV 的 chunk size=8 和 budget=1.56% 是权衡结果。该机制与所有基于 chunk-level sparse attention 的系统兼容。

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
