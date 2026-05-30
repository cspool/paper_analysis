## Data-Centric vs Model-Centric Parallel Configuration for MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Data-Centric 和 Model-Centric 是 HEXA-MoE 为不同 workload 规模提出的两种 MoE 分布式训练并行配置模式。核心区别在于通信的内容：Data-Centric 下各设备 all gather 完整 MoE 参数后本地计算（模型参数"移动"），适用于大规模 workload（batch size 大，参数通信开销被大计算量摊销）；Model-Centric 下各设备 all gather 数据批次后用本地参数 chunk 计算（数据"移动"），适用于小规模 workload（数据通信量小于参数通信量）。两种模式均使用 tensor parallelism（沿 FFN intermediate size 切分）替代 expert parallelism 的 all-to-all 通信。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
┌── Data-Centric vs Model-Centric 对比 ─────────────────────┐
│                                                              │
│ Data-Centric (适用: B×seq×top_k >> E×D_i×D_mid):            │
│   Forward:                                                   │
│     all_gather(W_shards) → 各设备持有完整 MoE 参数           │
│     ESMM(x, W_full, R(x)) → 本地完整计算                    │
│   通信: E×D_i×D_mid × 2 (W1+W2)                             │
│   计算: B×seq×top_k×D_i×D_mid (全部)                        │
│                                                              │
│ Model-Centric (适用: B×seq×top_k < E×D_i×D_mid):            │
│   Forward:                                                   │
│     all_gather(x) → 各设备持有完整数据                       │
│     ESMM(x_full, W_local, R(x)) → 各设备 1/N 输出           │
│     all_reduce(y_partial, SUM) → 聚合                       │
│   通信: B×seq×D_i (all_gather×2 + reduce×2)                 │
│   计算: B×seq×top_k×D_i×D_mid/N (每设备)                    │
│                                                              │
│ 关键特性:                                                    │
│ - 两种模式均为完全静态: workload 由 batch size 或            │
│   sub-dimension 精确决定                                     │
│ - 均使用 tensor parallelism (无 all-to-all)                  │
│ - 异构设备调度: B_i ∝ 1/t_i (data-centric) 或                │
│   h_i ∝ 1/t_i (model-centric)                               │
└──────────────────────────────────────────────────────────────┘
```

实验验证（Figure 7）：batch size 较小时 model-centric 延迟更低，batch size 较大时 data-centric 延迟更低。HEXA-MoE 配合 Pipeline-Shared Cache 解决 data-centric 的 backward 内存膨胀问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

通过 `data_centric` 参数切换。两种模式共享相同的 ESMM/ESS/ESTMM kernel 实现，差异仅在通信原语（all_gather vs all_reduce）和每设备的 workload 分配方式。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy
