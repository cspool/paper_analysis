## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoESys 提出了四项训练/推理系统优化策略：（1）**Hierarchical Storage（分级存储）**——将 MoE 模型的 sparse 参数（expert FFN 层）存储在 SSD/CPU 内存、dense 参数（attention 层）存储在 GPU HBM，通过理论公式（GPU-Node/CPU-Node/SSD-Node 内存约束方程）确定各存储层的容量分配；（2）**2D Prefetch Scheduling**——在 NVLink（水平维度）和 PCIe（垂直维度）上同时预取 dense 和 sparse 参数，与当前层的计算/通信重叠，使用类似 LFU 的 CPU cache 机制管理 sparse 参数的缓存命中；（3）**Elastic MoE Training**——根据各 task 的 batch size 动态调整计算节点数量（轻量 task 合并节点，重量 task 增加节点），消除负载不均造成的 "bubble" 空闲；（4）**Embedding Partition in Data Parallelism**——在 data parallelism 下对 embedding table 做 column-wise 切分（按 hidden_size 维度而非 vocab 维度），通过 3 次 AlltoAll 通信替代 AllReduce 同步，减少 memory footprint。
  - 实验比较：（1）large-scale MoE training：对比 DeepSpeed，在不同参数规模（13.9B-207.2B）、不同 GPU 数（8-128）下的 training throughput（tokens/s）和 GPU memory usage；（2）Elastic MoE Training：load imbalanced vs load balanced 配置下的 per-GPU throughput（samples/s），以及 UFO 模型上的 throughput 和 memory 对比 PyTorch v1.10；（3）Embedding Partition：不同 vocab/hidden/expert 配置下对比 baseline non-segment embedding 的 memory usage 和 speed；（4）Cross-wise comparison：各优化策略的 peak memory 和 computation speed 对比。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB（training），A100 40GB（部分 inference 实验）。单节点 8 GPU，通过 NVLink 互联；多节点通过 NIC + switch 互联。
  - Storage: HBM (GPU memory)、CPU DRAM、SSD、Intel Optane Persistent Memory（AppDirect 模式，FSDAX namespace，绕过 page cache 和 kernel 做 DAX 直接 load/store）。
  - Framework: PaddlePaddle / PaddleFleetX。

- 模型是什么。数据集和bench分别是什么。
  - 模型：GPT 系列 MoE 模型（参数 13.9B 到 207.2B，attention heads=64, hidden size=4096, vocab size=50304, layers=12, experts=8-128）；UFO（Unified Feature Optimization）视觉模型（12B sparse-gated MoE）；VIMER-UFO 2.0（billion-scale visual model）。
  - 数据集/benchmark：text generation 任务用于 MoE inference 评估；UFO 多任务训练（4 任务，batch sizes 512/256/128/128 模拟不平衡）。
  - 优化器：ADAMW，pure fp16 precision。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - MoESys 基于开源 PaddlePaddle/PaddleFleetX（https://github.com/PaddlePaddle/PaddleFleetX）实现。论文称 MoESys 代码将发布于 PaddlePaddle GitHub，截至搜索未找到独立 MoESys 仓库。
  - 算法 pipeline 核心——2D Prefetch + Hierarchical Storage 的执行流程：
    1. **参数分类**：MoE 模型参数分为 Dense（multi-head attention，始终激活）和 Sparse（expert FFN，选择性激活）。Dense 参数总量 D，Sparse 参数总量 S。
    2. **存储分配**（基于 ADAM optimizer，每个参数需 fp16 param + fp16 grad + fp32 master + fp32 momentum + fp32 variance = 16 bytes）：
       - GPU-Node 存储：全部 dense 参数状态 16D + 激活批次的 sparse 参数 4αS/L ≤ M_GPU × N
       - CPU-Node 缓存：高频 sparse 参数状态 16αS ≤ M_CPU × N
       - SSD-Node 全量：sparse 参数 master+动量+方差 12S ≤ M_SSD × N
       - 其中 α 为 sparse 参数激活概率（0<α<1），L 为 MoE 层数。
    3. **2D Prefetch**：水平维度（NVLink）→ AllGather 预取下一层 dense 参数（Algorithm 1），垂直维度（PCIe）→ 从 CPU cache 或 SSD 预取下一层 sparse 参数（Algorithm 2）。sparse 参数使用 hash table 记录命中频率（hits），CPU cache 满时淘汰最低命中频率且超过 threshold 的参数，使用 moving average 衰减（每 K step，hits × β）。
    4. **并行执行**：GPU 计算当前第 i 层 → 同时 NVLink 预取 dense 第 (i+1) 层参数 + PCIe 预取 sparse 第 (i+1) 层参数 → 下一层参数就绪无缝衔接。
  - Elastic MoE Training 流程：Gate network AlltoAll 收集 expert 选择结果 → 评估各 task workload 估算 → 合并轻量 task（combine nodes，比例 2:2）或拆分重量 task（add nodes，比例 1:1:1:1）→ 重分配 data partition → 同步参数。
  - Embedding Partition：embedding table [V, H] 沿 hidden_size 维度列切分 → 每个 worker 持有 [V, H/N] shard → Forward: AlltoAll 交换 input data → 本地 lookup → AlltoAll 交换结果 → Backward: AlltoAll 交换 gradients → 本地更新。无需 AllReduce。
