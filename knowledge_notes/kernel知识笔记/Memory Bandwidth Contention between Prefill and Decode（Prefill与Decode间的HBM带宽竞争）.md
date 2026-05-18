## Memory Bandwidth Contention between Prefill and Decode（Prefill与Decode间的HBM带宽竞争）

术语是什么？

在PD Multiplexing中，Prefill和Decode虽然使用不同的SM分区和CUDA Stream并行执行，但它们共享同一GPU的HBM（高带宽内存）带宽和L2 Cache。当两阶段同时访问HBM时，产生HBM带宽竞争（Memory Bandwidth Contention），导致各自性能下降。Prefill是计算密集型（compute-bound），Decode是内存密集型（memory-bound），因此Decode对HBM带宽竞争更为敏感。

论文Table 2给出两阶段不同场景的compute complexity分析（d=hidden dimension, L=总token长度, n=new context长度, r=reused context长度）：

| Phase | Attention复杂度 | FFN复杂度 |
|-------|----------------|----------|
| Prefill (无cache) | O(L²d + Ld²) | O(Ld²) |
| Prefill (有cache) | O(nd²) | O(nd²) |
| Decode | O(d² + (r+1)d) | O(d²) |

该分析揭示：Prefill的attention对总context长度呈二次复杂度（无cache时），Decode的attention对reused context呈线性复杂度。当reused context极长时（multi-turn场景可>50K tokens），Decode需大量读取KV cache，与同时执行的Prefill竞争HBM带宽，导致Decode显著变慢。

从kernel调度角度拆解术语：

伪代码表示带宽竞争的kernel执行交互：

```
// 两个CUDA Stream并行执行
Stream_Prefill (GreenContext_Prefill):  // 更多SM
  for each prefill_block:
    // 计算密集：大量矩阵乘法和注意力计算
    QKV = matmul(input, W_qkv)         // HBM读取W_qkv权重
    attn_out = flash_attention(Q, K, V) // HBM读写KV Cache
    ffn_out = matmul(attn_out, W_ffn)  // HBM读取W_ffn权重

Stream_Decode (GreenContext_Decode):    // 少量SM
  for each decode_iteration:
    // 内存密集：主要瓶颈在KV Cache和权重的HBM访问
    q = matmul(token, W_q)             // HBM读取W_q (<1% 计算量)
    attn_out = paged_attention(q, KV)  // HBM读取KV Cache (主要瓶颈)
    ffn_out = matmul(attn_out, W_ffn)  // HBM读取W_ffn权重

// HBM带宽竞争发生在：
// - Prefill的W_qkv读取 vs Decode的KV Cache读取
// - Prefill的W_ffn读取 vs Decode的W_ffn读取
// - L2 Cache行冲突（特别是attention的KV页访问模式与matmul的连续访问模式冲突）
```

从kernel执行角度看：
1. Decode的每token计算量极小（~2×参数量 FLOPs），但HBM访问量很大（读取所有权重+KV Cache页面）。
2. Prefill的每token计算量大（批量矩阵乘），HBM访问模式为连续流式访问。
3. 两者并行时，DRAM控制器需要在两种访问模式间仲裁，L2 Cache的命中率因竞争而下降。
4. 结果是Decode的迭代延迟显著增加（因为Decode对带宽更敏感），可能导致ITL SLO违规。

术语一般如何实现？如何使用？

MuxWise通过Contention-tolerant Estimator来建模和量化这种带宽竞争：离线profile不同SM分配下的Decode延迟曲线和Prefill吞吐曲线，在线性回归基础上加入竞争因子，实时预测给定SM分配下的实际ITL。当预测ITL接近SLO时，调度器减少Prefill的SM分配或降低Prefill并发度。Bullet使用SM-scaling Roofline Model (SRM)作为替代建模方法：基于roofline分析建模compute/memory/network三个维度的性能上界随SM数量变化的关系，用稀疏concurrent sample校准（<1小时 vs Estimator的~12小时），通过roofline边界预测而非全网格查表。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

---

