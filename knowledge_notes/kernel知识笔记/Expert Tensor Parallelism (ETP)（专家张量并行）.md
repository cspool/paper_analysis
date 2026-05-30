## Expert Tensor Parallelism (ETP)（专家张量并行）

术语是什么？
Expert Tensor Parallelism (ETP) 是 Tensor Parallelism (TP) 在 MoE 层的特定形式。与标准 TP 切分 Attention 层的 hidden dimension 不同，ETP 在 MoE 的 expert FFN 内部进行张量切分——每个 expert 的权重矩阵沿 hidden/intermediate dimension 被切分到多个 GPU 上。ETP 与 EP 构成二维网格 [EP, ETP]，即 experts 先按 EP 分布式放置，每个 expert 内部再按 ETP 切分。

从 kernel 调度角度拆解术语：
在 MoE Parallel Folding 框架中，MoE 层的 ETP 通信流程（以 ETP=2, EP=2, 4 GPU 为例）：

```
Forward Pass:
1. Router: 本地 token → expert assignment + permutation
2. All-to-All-V (跨 EP 组): token 发送到对应 expert 所在 rank
3. AllGather-V (跨 ETP 组): ETP 组内广播，确保所有 rank 持有完整 activation
4. Expert GEMM: 各 rank 计算其分配的 weight partition
5. ReduceScatter-V (跨 ETP 组): 聚合分发输出 hidden states
6. All-to-All-V (跨 EP 组): token 返回原始 rank
7. Unpermutation: 恢复 token 顺序

Backward Pass: AG/RS 互换为 RS/AG
```

通信量对比：
- **ETP 通信**：AllGather + ReduceScatter = 2 × bsh (n-1)/n，通信量与 TP 相同
- **EP 通信**：2 × All-to-All = 2 × (k/n) × bsh (n-1)/n，其中 k 为 top-k
- 当 k < n 时，EP 通信量小于 ETP；但 fine-grained MoE 中 k 大且 expert hidden size 小，ETP 通信占比可达 70%+

术语一般如何实现？如何使用？
- ETP 在 Megatron-Core 中通过 moe_groups["TP"] 实现，其 degree = etp
- 当需要将大 expert 切分到多 GPU 以减少单 GPU 内存压力时使用 ETP
- MoE Parallel Folding 允许将 ETP 替换为 EP（设置 etp=1），将通信从 AG/RS 转为 A2A，对 fine-grained MoE 特别有效
- 实现使用 NCCL AllGather-V 和 ReduceScatter-V 集合通信

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
