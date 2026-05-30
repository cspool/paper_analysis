## Co-scheduling Intra-node and Inter-node Communications

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Co-scheduling Intra-node and Inter-node Communications 是 FSMoE 的核心系统创新——将节点内通信（ESP-AllGather/ESP-ReduceScatter，NVLink）和节点间通信（AlltoAll，InfiniBand）在流水线中协同调度。利用两者物理网络隔离（NVLink 900GB/s vs InfiniBand 100GB/s on DGX H100），在不同数据 chunk 上并行执行。当 MP 和 ESP group 对齐节点内 GPU 数时，此优化自动生效。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

协同调度流水线（backward, r=4）：
```
Chunk 0: AG→A2A→RS→Exp
Chunk 1:    AG→A2A→RS→Exp    (C1.AG 与 C0.A2A 重叠)
Chunk 2:       AG→A2A→RS→Exp (C2.AG 与 C1.A2A 重叠)
Chunk 3:          AG→A2A→RS→Exp→GAR
```

Table 2 显示通信占总训练时间 >50%，FSMoE vs FSMoE-No-IIO 额外加速约 5-6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

依赖 NCCL CUDA stream 机制——不同 chunk 的节点内/节点间通信在不同 stream 异步执行，GPU 调度器自动并行。限制：MP/ESP group 必须对齐节点内 GPU 数。在 Testbed-A（N_MP=N_ESP=8）和 Testbed-B（N_MP=N_ESP=4）上验证，对 Mixtral-8x7B 等大模型训练尤其有效。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
