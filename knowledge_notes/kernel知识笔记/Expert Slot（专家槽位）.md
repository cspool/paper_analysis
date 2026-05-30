## Expert Slot（专家槽位）

术语是什么？
Expert Slot 是 SmartMoE 提出的用于统一表达 MoE 模型混合并行策略的核心抽象。它将每个 GPU worker 上的 expert 子网络存储表示为一个"槽位"（slot），通过三个属性统一描述任意 DP/TP/PP/EP 组合：(1) 每个 slot 的容量（capacity）：0 到 1 的分数，表示存储完整 expert 还是部分 expert；(2) 每个 worker 的 slot 数量（#slots）：正整数；(3) 每个 worker 的 MoE 层数（#layers）。例如纯 EP 配置为 capacity=1, #slots=E/N, #layers=L；EP+TP 配置为 capacity=1/T, #slots=T×E/N, #layers=L；EP+PP 配置为 capacity=1, #slots=E/(N/P), #layers=L/P。

从kernel调度角度拆解术语：
以 (L=2 MoE layers, E=4 experts, N=4 GPUs) 为例的 slot 配置：
```
纯 EP:
  GPU_0: [slot0(E0, cap=1), slot1(E1, cap=1)]
  GPU_1: [slot0(E2, cap=1), slot1(E3, cap=1)]
  #slots=1, #layers=2 per GPU（per-layer 各一个 slot）

DP=2 + EP:
  GPU_0: [slot0(E0, cap=1), slot1(E2, cap=1)]
  GPU_1: [slot0(E1, cap=1), slot1(E3, cap=1)]
  #slots=1, #layers=2 per GPU

TP=2 + EP:
  GPU_0: [slot0(E0_half0, cap=0.5), slot1(E1_half0, cap=0.5)]
  GPU_1: [slot0(E0_half1, cap=0.5), slot1(E1_half1, cap=0.5)]
  #slots=2, #layers=2 per GPU
  cap=0.5 表示每个 slot 存储 expert 参数的一半（另一半在 TP partner GPU）
```

Expert Slot 抽象的关键作用：它使不同混合并行方案之间可以互相比较和转换——相同 slot 配置意味着切换时不需内存分配/释放（只交换参数），这定义了 SmartMoE "pool" 的边界。

术语一般如何实现？如何使用？
SmartMoE 在 FastMoE 框架上实现 expert slot 抽象：运行时，expert 到 slot 的映射由 expert placement 算法动态决定；slot 的 capacity/#slots/#layers 属性由离线阶段搜索的混合并行策略固定。切换时，只有受影响的 slot 内的 expert 参数通过 All-to-All 重新分配。安装：`cd src/fastmoe && USE_NCCL=1 python setup.py install --user`。

涉及论文标题：
- SmartMoE Efficiently Training Sparsely-Activated Models through Combining Offline and Online Parallelization
