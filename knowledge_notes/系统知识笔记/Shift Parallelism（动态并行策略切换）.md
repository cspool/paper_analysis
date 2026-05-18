## Shift Parallelism（动态并行策略切换）

术语是什么？

Shift Parallelism是Snowflake AI Research提出的动态多GPU并行策略切换机制：在同一个LLM推理部署中保留两套并行配置——base configuration（SP或mixed SP+TP，偏向大batch的吞吐和TTFT优化）与shift configuration（full TP，偏向小batch的TPOT优化），运行时根据当前batch token数是否超过shift threshold来动态选择执行路径。核心insight是Sequence Parallelism (SP)与Tensor Parallelism (TP)共享相同的KV cache attention head layout（KV Cache Invariance），使同一请求在SP与TP之间切换时无需搬移KV cache。

Shift Parallelism在vLLM推理framework中以插件形式（ArcticInference, GitHub: snowflakedb/ArcticInference, Apache-2）实现。base model和shift model各自编译并capture CUDA graphs，运行时根据`batch_size > shift_threshold`的条件选择执行base（Algorithm 1[SP, TP]）或shift（Algorithm 1[1, SP×TP]）。两套model共享attention layer和KV cache，shift model额外权重开销约为`1/SP`（如SP=8时~12.5%）。

从系统架构角度拆解术语：

Shift Parallelism在系统架构层的运作流程：

1. **请求到达与调度**：请求到达后进入vLLM continuous batching队列，scheduler按PagedAttention管理KV cache页表。

2. **阈值决策**：每个forward iteration前，runtime统计当前batch中的token总量n。若`n > shift_threshold`（默认256），选择base config偏向吞吐；否则选择shift config偏向延迟。这是Algorithm 2的核心逻辑。

3. **Base path（大batch→SP/混合）**：使用Algorithm 1[SP, TP]执行。输入token按SP degree沿sequence维度切分到不同SP rank→QKV projection→SP group内fused all-to-all将attention head shard聚合到对应GPU→本地attention→SP all-to-all返回sequence layout→O projection→TP all-reduce（MLP路径）→输出。

4. **Shift path（小batch→Full TP）**：使用Algorithm 1[1, SP×TP]执行。full TP方式并行整个batch，column/row partitioning for attention and MLP。避免SP在小batch时的padding/load imbalance（batch size=9, SP=8时效率仅50%）。

5. **KV cache共享**：由于base和shift使用相同的attention head ordering（通过SP_TP group维护），KV cache在两种配置间共享不搬迁。请求在SP prefill和TP decode之间切换时，已有的K/V历史状态直接复用。

6. **生产环境组合**：论文将Shift Parallelism与SwiftKV（prefill优化）、speculative decoding组合，在单一部署中同时实现低延迟和高吞吐。报告相对最佳throughput-optimized部署有3.4×更低completion time和1.06×更高throughput。

术语一般如何实现？如何使用？

通过vLLM插件系统实现。使用方式：

```bash
vllm serve <model> \
    --tensor-parallel-size 1 \
    --ulysses-sequence-parallel-size 4 \
    --enable-shift-parallel \
    --shift-parallel-threshold 512
```

关键约束：`SP × TP = 总GPU数`以确保KV cache invariance。`--shift-parallel-threshold`控制切换阈值。lower threshold倾向于低延迟（更频繁使用TP），higher threshold倾向于高吞吐（更频繁使用SP）。论文中使用8×H200 GPU，典型配置为base (SP=4, TP=2)，shift (SP=1, TP=8)。在Llama-70B-FP8、4k input/250 output场景下，Shift Parallelism的median TTFT 148ms、median TPOT 51ms、peak throughput 69,147 tok/s。

涉及论文标题：
- Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads
