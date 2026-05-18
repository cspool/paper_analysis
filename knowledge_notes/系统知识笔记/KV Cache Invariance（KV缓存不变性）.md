## KV Cache Invariance（KV缓存不变性）

术语是什么？

KV Cache Invariance是Shift Parallelism论文的核心观察性质：Tensor Parallelism (TP)与Sequence Parallelism (SP/Ulysses)在特定条件下共享相同的attention head分布和KV cache memory layout。当`SP × TP = P`（P为总GPU数）时，attention heads在GPU间的分布（每个GPU持有哪些heads）和head ordering（heads在GPU内的排列顺序）可以保持一致，使同一请求在SP和TP两种并行策略间切换时无需搬移或重组已有KV cache。

论文发现head ordering的invariance并非自动成立。对于任意(SP, TP)组合（如SP=3, TP=2），base config的SP groups会产生interleaved head ordering（如(0,2,4,1,3,5)），而shift config（TP=6）按rank顺序的直接分布是(0,1,2,3,4,5)。解决方案是构造SP_TP group：`SP_TP = [[0, 2, 4, 1, 3, 5]]`，让shift model按base config的SP group order加载权重，维持KV cache coherence。

从系统架构角度拆解术语：

KV Cache Invariance在请求生命周期中的作用：

1. **并行策略切换场景**：一个请求首先在base config（SP=4, TP=2）下完成prefill。SP将input sequence切分到4个SP ranks，attention时通过all-to-all切换到head parallel layout，各GPU处理自己负责的attention heads并写入KV cache。

2. **切换发生**：当流量下降、batch token数 ≤ threshold时，runtime切换到shift config（SP=1, TP=8）。同一请求进入decode阶段。

3. **KV cache读取一致性**：在shift config（full TP=8）下，每个GPU需读取对应attention heads的KV cache条目。由于base config已固定head ordering（通过SP_TP group锁定interleaved order），shift config加载的weight shard与base config的KV cache head布局一致。GPU 0在两种配置下都負責相同的attention heads，无需跨GPU搬移K/V数据。

4. **与TP↔DP不兼容的对比**：DP的KV cache layout与TP不同（DP每个GPU持有完整模型副本处理不同请求，K/V在不同请求/GPU间不保证head一致性），因此TP↔DP切换需要昂贵的KV cache搬迁或重新计算。Shift Parallelism通过选择SP（而非DP）作为throughput-optimized配置，利用SP与TP的KV cache layout一致性规避了这一问题。

术语一般如何实现？如何使用？

实现依赖于SP_TP group的定义。以8 GPU、(SP=3, TP=2)为例：

```
TP groups:  [[0, 1], [2, 3], [4, 5]]
SP groups:  [[0, 2, 4], [1, 3, 5]]
SP_TP group: [[0, 2, 4, 1, 3, 5]]  # Span both dims, SP group order
```

在separate models实现中，base model按(SP, TP) group加载权重，shift model按SP_TP group加载权重。论文也考虑了on-the-fly weight slicing替代方案（从base model weight partition中切片），但由于Hopper FP8 tensor core限制需矩阵转置，性能不如separate models。用户无需手动配置invariance——`--enable-shift-parallel`和`--shift-parallel-threshold`由ArcticInference插件自动处理SP_TP group构建。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

---
