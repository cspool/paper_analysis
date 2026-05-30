## Hierarchical Storage for MoE Training (分级存储 / 异构存储管理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Storage for MoE Training 是 MoESys 提出的 MoE 模型分布式训练的异构存储管理策略。其核心思想是将 MoE 模型参数按激活特性分为两类——dense 参数（multi-head attention 层，始终激活，参数总量 D）和 sparse 参数（expert FFN 层，选择性激活，参数总量 S）——并分别存储在不同的存储介质上：dense 参数常驻 GPU HBM（避免频繁数据搬运），sparse 参数存于 SSD/CPU 内存（因其规模远超 GPU 容量，且仅在按 α 概率激活时需要）。引入 Intel Optane Persistent Memory (AppDirect + DAX) 替代传统 SSD，以解决 SSD 擦除寿命和 PCIe 延迟问题。MoESys 给出 GPU-Node、CPU-Node、SSD-Node 三层存储的容量约束公式（以 ADAM optimizer 为例：每个参数需 fp16 param + fp16 grad + fp32 master + fp32 momentum + fp32 variance = 16 bytes），其中 α 为 sparse 参数激活概率（0<α<1），L 为 MoE 层数，N 为设备数：GPU-Node 存储 16D+4αS/L ≤ M_GPU·N；CPU-Node 缓存 16αS ≤ M_CPU·N；SSD-Node 全量 12S ≤ M_SSD·N。

从系统架构角度拆解术语：
Hierarchical Storage 在 MoESys 的训练全流程中扮演存储引擎角色：
1. **参数分类阶段**：训练开始时，按 MoE layer 结构将模型参数分为 dense 组和 sparse 组。
2. **存储分配阶段**：dense 参数（含 optimizer states）加载到 GPU HBM 并常驻。sparse 参数（expert FFN 权重）的全部 optimizer states 写入 Optane PMem (SSD-Node)，仅训练中按 expert 选择结果按需加载到 GPU。
3. **训练循环中的存储交互**：每个 training step，Gate 网络决定哪些 expert 被激活 → CPU cache (LFU 机制) 查询所需 sparse 参数 → cache hit 则从 CPU→GPU 传输，cache miss 则 SSD→CPU→GPU 传输 → 计算完成后 sparse 参数梯度回传。
4. **Cache 管理**：CPU memory 作为 sparse 参数的热数据缓存，使用 hash table 记录每个 sparse 参数的历史命中次数 (hits)，按照 LFU 策略（淘汰最低命中频率且超过 threshold 的参数）管理缓存换入换出，每 K step 对 hits 做 moving average 衰减。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现依赖多级存储硬件的带宽层次：GPU HBM (2TB/s, 80GB per A100) → NVLink (900GB/s) → CPU DRAM (DDR4 ~200GB/s) → PCIe (32GB/s per direction) → Optane PMem / SSD。
- MoESys 使用 Intel Optane PMem 的 AppDirect 模式（而非 Memory 模式），设置 namespace 为 FSDAX，利用 Ext4 DAX 特性实现绕过 page cache 和 kernel 的直接 load/store 操作，避免中断和上下文切换开销。
- 与传统 ZeRO-Infinity（将所有参数统一 prefetch，不区分 sparse/dense）相比，Hierarchical Storage 的核心改进是认识到 MoE 中 sparse 参数和 dense 参数的异构访问模式——dense 参数每次 step 都需要、sparse 参数仅 α 概率被访问——因此对两者采取不同的存储和预取策略。
- 该策略使得 MoESys 在 104.1B MoE model (64 A100 GPUs) 训练中 GPU memory 消耗从 DeepSpeed 的 66.3GB 降至 54.4GB（降低约 12GB per GPU）。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
