## Online Upcycling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Online Upcycling 是 NVIDIA NeMo 框架中实现的分布式 Sparse Upcycling 机制，允许用户在训练启动时直接指定 dense checkpoint 路径和并行训练配置，框架自动完成 dense→MoE 的权重转换和分布式初始化，无需手动预处理或离线脚本。传统 upcycling 的挑战在于：dense checkpoint 可能包含数十亿参数，upcycling 后总参数量激增（如 8B→34.4B），单节点无法承载完整模型。Online Upcycling 通过以下设计解决：(1) 先按并行训练配置分片 dense checkpoint，(2) 各设备独立对本地分片进行 upcycling（复制 FFN 权重、初始化 router），(3) 消除跨设备权重复制，避免额外通信开销。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Online Upcycling 的分布式执行流程：

```
1. 用户提供: dense checkpoint 路径 + 并行配置 (TP, EP, PP, CP, DP)

2. NeMo 训练启动:
   ┌─────────────────────────────────────────────────────────┐
   │ Step 1: Shard dense checkpoint                         │
   │   按 PP 切分层 → 按 TP 切分张量 → 按 EP 分配 expert   │
   │   每个 GPU 仅加载其本地所需部分                         │
   │                                                        │
   │ Step 2: Per-device Upcycling                           │
   │   GPU_i 对本地 MoE layer:                              │
   │     - 复制本地 FFN 权重 N 次 → N 个 expert 的本地参数  │
   │     - 随机初始化 router 的本地分片                      │
   │   GPU_i 对本地 non-MoE layer:                          │
   │     - 直接复制 attention/norm/embedding 权重            │
   │                                                        │
   │ Step 3: 开始训练                                       │
   │   All-to-All dispatch → Expert compute → Combine       │
   │   无需额外跨设备权重复制                                │
   └─────────────────────────────────────────────────────────┘
```

关键设计优势：
- 内存高效：每个设备仅处理本地分片，总内存需求 = 单设备 MoE 模型大小，而非完整 MoE checkpoint
- 通信高效：无跨设备权重复制步骤，upcycling 在数据并行域内独立完成
- 用户友好：用户仅需提供 dense checkpoint 路径，框架自动并行化处理

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现于 NVIDIA NeMo (https://github.com/NVIDIA/NeMo)，基于 Megatron-Core 分布式训练后端：
- 作为 NeMo 训练脚本的启动参数：`--upcycle-from-checkpoint /path/to/dense/checkpoint`
- 配合并行配置：`--tensor-model-parallel-size 2 --expert-model-parallel-size 8 --pipeline-model-parallel-size 4`
- 支持任意 dense LLM checkpoint（需与目标 MoE 架构兼容）
- 适用框架 PyTorch + Megatron-Core + NCCL 通信后端

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling
