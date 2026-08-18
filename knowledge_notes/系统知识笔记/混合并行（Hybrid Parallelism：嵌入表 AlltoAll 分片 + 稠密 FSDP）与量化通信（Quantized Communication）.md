## 混合并行（Hybrid Parallelism：嵌入表 AlltoAll 分片 + 稠密 FSDP）与量化通信（Quantized Communication）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 混合并行（hybrid training / hybrid parallelism）是大型推荐模型（LRM）的标准分布式训练方式（DLRM 论文 [50] 提出）：稀疏嵌入表（占模型大部分参数，存离散特征如 user/item 的稠密向量）用 embedding sharding 跨 GPU 分片，经 AlltoAll 集合通信交换（每 batch 每特征查表后把"该发往各 rank 的嵌入"整体交换）；稠密组件（MLP、交互模块、预测层）用 Fully Sharded Data Parallel（FSDP，见"Model-Sharded Data Parallelism"条目）分片。这样嵌入与稠密两套并行策略在同一训练作业中共存，故名"混合"。
- 量化通信（quantized collective communications，[73]）把集合通信中传输的张量先量化再传（如 BF16 下通信），降低通信带宽占用。LoKA 为公平对比，对所有模型统一启用 BF16 量化通信，与计算 datatype 无关。
- LoKA 视角下的角色：LRM 训练是通信密集型环境（嵌入 AlltoAll + 稠密 FSDP AllGather/ReduceScatter），低精度只作用于稠密计算，通信占比高时低精度收益被稀释——因此 LoKA 的端到端加速与"计算-通信比"强相关：小规模高计算强度模型受益最大（Wukong 1.19× 训练），更大规模/更多 GPU 时收益随通信占比上升而下降（256 GPU 仍 +10%）；迭代级剖析（Fig.11）显示 LoKA 加速主要来自 GEMM 延迟约 2× 降低，稀疏嵌入 pipeline 与 GPU 间通信模式不变（稀疏通信的微小差异归因于 run-to-run 波动）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一次 Wukong 训练迭代（64 GPU，混合并行）：① 嵌入侧——每个 batch 的稀疏特征按本地 embedding 表查表得嵌入向量，按目标 rank 分桶 → AlltoAll 把嵌入交换到对应 rank（每 rank 收到全局 batch 中自己负责的嵌入片段）→ 拼接成稠密特征；② 稠密侧——FSDP：前向逐层 AllGather 参数分片 → GEMM（LoKA 在此层做 FP8 + BlockNorm/Hard Swish）→ 反向 ReduceScatter 梯度分片；③ 通信量化——AllGather/ReduceScatter 与 AlltoAll 传输在 BF16 量化后发送（与计算 FP8 无关）；④ LoKA Dispatch 的 kernel 选择只改稠密计算路径，不改嵌入与通信拓扑。
- 与计算-通信比的关系：小 batch/小集群（Wukong 24 GFLOPs/sample，H100 6K batch 32 GPU）计算占比高，FP8 GEMM 加速显著；大模型（ELFM 1343B，2K batch 256 GPU）通信占比高，加速比下降但仍为正。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：TorchRec（PyTorch 推荐系统域库）提供嵌入表分片与 AlltoAll 的混合并行训练支持；平衡分片（balanced sharding，[75]）均衡各 GPU 嵌入负载；FSDP（PyTorch 内置）负责稠密分片。量化通信实现见 quantized collective communications（KDD'20）；LoKA 统一在 BF16 使能。使用场景：工业级广告/推荐 ranking 模型的在线持续训练（LRM 需随用户偏好/新 item 流式更新），训练与推理性能都关键。别名：DLRM 混合训练 / hybrid training。

MTIA 300 补充视角（ISCA'26，DLRM 训练的混合并行与通信模式）：MTIA 300 的 DLRM 训练沿用混合并行——dense 层数据并行（AllReduce 同步梯度）、sparse 层 embedding 表按 table-wise/row-wise 模型并行分片（AllToAllv 交换特征与结果）、分布式 Shampoo 优化器加 AllGather（优化阶段）；embedding 表因超单卡容量（150B 参数 99% 在稀疏侧）必须分片。与 GPU 差异：collective 由 HCCL 卸载到 ME/NMC（而非 NCCL 主机驱动）；关闭 row-wise FP8 量化通信（MTIA 300 上走低效 RISC-V，关闭后 +4.4%）；大 HBM 支撑 local batch 10240（24 卡、Perf/TCO 1.42× vs H100 40 卡 6144）。40 卡迭代通信画像：AllReduce 入站 1.6 GB、AllGather 2.1 GB、35 次 AllToAllv（1 KB-1 GB 可变消息），整体通信性能超 H100 3.9×。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
