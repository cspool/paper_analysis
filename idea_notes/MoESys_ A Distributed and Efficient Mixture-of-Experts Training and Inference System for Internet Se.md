## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- baseline方法是什么？
  Baseline 是 DeepSpeed-MoE 的训练和推理系统。DeepSpeed 使用 ZeRO 策略 + 参数预取实现 MoE 训练，但存在三个核心缺陷：
  (1) **存储管理粗粒度**：DeepSpeed 的 Zero-Infinity 将参数统一 prefetch，不区分 MoE 中 sparse parameter（expert FFN，选择性激活且占存储大头）和 dense parameter（attention，始终激活）的异构特性，导致 SSD 寿命损耗和性能下降（SSD 满容量时性能衰减）；
  (2) **通信效率受限**：DeepSpeed 的 AlltoAll 设计主要通过层间 tensor fusion 将小 packet 合并为大 packet 通信，解决了 per-port 通信量小的问题，但未针对实际集群网络拓扑（intra-node NVSwitch vs inter-node switch hierarchy）做优化，跨 rank 通信经过 spine switch 造成路由冲突和带宽浪费；
  (3) **负载均衡盲点**：在 multi-task MoE 训练（如 UFO）中，不同 task 的 batch size 差异导致"木桶效应"——重 task 节点处理时间远长于轻 task 节点，轻 task 节点计算完毕后空闲等待（bubble），整体 FLOPS 利用率低。
  全栈执行例子：训练一个 100B+ MoE 模型 → 每层 MoE 含 attention (dense) + MoE FFN (sparse, 64 experts) → DeepSpeed 通过 Zero stage 3 对所有参数做统一 partition 和 prefetch → "Forward: 从 SSD/CPU 预取所有参数 → GPU compute → AlltoAll exchange expert hidden states → Backward → AlltoAll sync gradients → Optimizer update" → 问题 1：AlltoAll 通信中 GPU0 of Node1 (cluster A, rank0) 与 GPU7 of Node2 (cluster B, rank7) 间的数据经过路径 NIC1→LE1→SPq→LE1→NICn，spine switch 成为瓶颈且与其它 GPU pair 的通信竞争，形成阻塞。问题 2：SSD 频繁写入导致擦除次数高，且不区分 sparse/dense 使得大量 dense 参数预取也走 PCIe，浪费 NVLink 高带宽。问题 3：UFO 多任务训练时 task1（batch 512）的 GPU 需要 2× 的 batch 数据，处理时间远长于 task3（batch 128），其余 GPU 等待。编译框架/Kernel调度：论文未明确说明 baseline 的 kernel 细节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MoESys**，一个基于 PaddlePaddle/PaddleFleetX 的 MoE 训练与推理系统，通过四项核心设计解决 DeepSpeed 的三个缺陷：

  **对应缺陷 1（存储管理粗粒度）→ Hierarchical Storage + 2D Prefetch Scheduling**
  - 将 MoE 参数按激活特性分为 sparse（expert FFN）和 dense（attention）：dense 参数始终在 GPU HBM；sparse 参数存 SSD，通过 CPU memory 做 LFU 缓存。
  - 2D Prefetch：水平维度利用 NVLink（高带宽）预取 dense 参数 → AllGather 获取所有 shard；垂直维度利用 PCIe 从 CPU cache 或 SSD 预取 sparse 参数。两个维度并行，与当前层计算重叠。
  - 引入 Intel Optane PMem (AppDirect + DAX) 替代传统 SSD，提供字节级寻址 + DRAM-like 延迟 + SSD-like 持久性，解决传统 SSD 的擦除寿命和延迟问题。
  - GPU-Node/CPU-Node/SSD-Node 的容量约束公式确保各存储层不溢出。

  **对应缺陷 2（通信效率受限）→ Hierarchical AlltoAll (Resource-Aware Communication)**
  - 利用网络拓扑层次：阶段一 intra-node AlltoAll via NVSwitch → 将数据搬移到同节点内对应 rank 的 GPU；阶段二 inter-node AlltoAll via NIC grouped by rank → 同 rank GPU 直连同一 leaf switch，不经过 spine switch。
  - 效果：peer-to-peer 通信效率提升 p 倍（p=单节点 GPU 数），inter-node 带宽利用率最大化。80.7B model / 4 nodes 32 GPUs 下通信阶段 speedup 15.5%，整体训练提升 10.3%。

  **对应缺陷 3（负载均衡盲点）→ Elastic MoE Training**
  - 动态调整节点数：轻量 task 合并节点（2 task→1 node），重量 task 拆分节点（1 task→多 node + data parallelism）。
  - 成本感知的 scale up/down 策略：upscaling 提升整体 throughput，downscaling 在资源受限时控制成本。
  - UFO model / 4 tasks 下 per-GPU throughput 提升 18.2%；VIMER-UFO 2.0 上 throughput 提升 64%，memory 降低 18%。

  **额外优化：Embedding Partition in Data Parallelism + Ring Memory Offloading**
  - Embedding Partition：沿 hidden_size 列切分（非 vocab 维度），3 次 AlltoAll 替代 AllReduce，大幅降低大 vocabulary 场景下的 GPU memory（如 400M param 配置从 15.81 GB 降至 8.63 GB）。
  - Ring Memory Offloading (inference)：CPU-GPU 环形内存流水线——GPU 缓存 K 份 expert 参数，计算第 i 层时释放 Pi 并异步加载第 (K+i) 层，多个 CUDA stream 实现 compute 与 data movement 重叠，GPU memory 节省 ≥30%。

  全栈执行例子（对比 baseline）：训练 104.1B MoE model on 64 A100 GPUs → 
  - **算法pipeline层**：参数分类后 dense 16D=16×1B≈16GB per device 常驻 GPU HBM；sparse 12S≈12×103B≈1236GB 存 SSD，α=0.02 激活概率下 GPU 仅需 4αS/L≈0.7GB 的 sparse 参数空间。
  - **系统框架层**：PaddleFleetX 分布式训练 → data parallelism (dense) + expert parallelism (sparse) → Gate 网络 AlltoAll 收集路由结果 → 2D prefetch 同时从 NVLink 预取下一层 dense 参数 + 从 CPU/SSD 预取下一层 sparse 参数（LFU cache 命中检查 → 未命中则 SSD→CPU→GPU）。
  - **编译框架层**：PaddlePaddle JIT 转静态图 → graph fusion 消除冗余 → kernel fusion (fused MHA) 减少 kernel launch 开销。
  - **Kernel调度层**：Hierarchical AlltoAll 替代标准 AlltoAll：intra-node 900GB/s NVSwitch → inter-node 100G NIC 同 rank 分组的 leaf switch 直连 → 通信耗时减少 15.5%。Custom H2D/D2H kernel：cudaHostAlloc pinned memory + cudaMemcpyAsync 异步传输与计算重叠。
  - **硬件架构层**：A100 GPU (108 SM, 80GB HBM2e, 2TB/s) + NVSwitch (900GB/s) + Mellanox 100G NIC + leaf/spine 交换机拓扑。论文未涉及 RTL/芯片/模拟器。

  效果：training throughput 209970 tokens/s vs DeepSpeed 157728 tokens/s（+33%），memory 54.4GB vs 66.3GB（-18%）。编译框架/芯片设计：论文未明确说明。

- baseline方法是什么？
  Baseline 是传统的 Speculative Decoding（SD）应用于 MoE 模型。学术界普遍认为 SD 对 MoE 无效，原因有二（1）对于小 batch：验证阶段多 draft token 激活更多 expert，导致参数加载量显著增加，T_T(B,γ) 远大于 T_T(B,1)，SD speedup 很低甚至 <1.0；（2）对于大 batch：系统进入 compute-bound 状态，验证时间随 token 数线性增长 T_T(B,γ)/T_T(B,1)→γ，SD 同样失去加速效果。现有 SD 研究主要关注提升 acceptance rate α（算法指标），忽视了目标模型架构和 workload 等系统因素对 speedup 的影响。
  全栈执行例子：1 个 request → Draft model 自回归生成 4 个 draft tokens → Target MoE model（如 Mixtral-8x7B，K=2）以 batch=1, γ=4 验证 4 个 tokens：4×2=8 次 expert 激活，可能激活 5-8 个不同 experts → 加载 5-8 个 expert 的权重（每个约数百 MB）→ 由于 batch=1 时仅需激活 2 个 expert 做单 token 解码，验证的参数加载量变为原来的 2.5-4× → T_T(B,γ) >> T_T(B,1) → SD speedup 远低于 dense 模型。Serving 调度层：vLLM continuous batching 下请求数少时无法隐藏 expert 加载延迟。编译框架/Kernel调度/硬件架构：论文未明确说明。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法推翻"SD 对 MoE 无效"的固有认知，证明在**中等 batch size**（所有 expert 已激活但 GPU FLOPs 未充分利用）下，SD 对 MoE 的加速效果甚至优于 dense 模型。核心设计：
  1. **中等 batch size 消除额外 expert 加载**：当 batch size B 足够大使 N(B)≈E（所有 expert 已被单步解码激活），验证 B×γ 个 tokens 不再激活新 expert，仅增加计算量 → 系统处于 memory-bound 时计算增量几乎免费 → T_T(B,γ)≈T_T(B,1)。
  2. **更稀疏的 MoE 放大加速窗口**：Texp(t;ρ) = ρt/(1-(1-ρ)^t)，ρ 越小 → 每个 expert 处理 token 越少 → 系统更 memory-bound → 延迟 compute-bound 转型 → SD 有效加速的 batch size 范围更广。
  3. **Target Efficiency（新系统指标）**：= T_T(B,1)/T_T(B,γ)，解耦系统瓶颈与 acceptance rate 等算法指标。即使 acceptance rate 相同，target efficiency 也能解释不同模型/工作负载下 SD speedup 的巨大差异。
  4. **性能模型（Algorithm 1）**：融合 roofline model（G(t)）、激活专家数 N(t)、expert load Texp 三个因子，通过最小二乘拟合确定参数，预测任意 workload 的 SD speedup。
  全栈执行例子（对比 baseline）：B=32 个 requests → Draft model 生成 γ=4 个 draft tokens → Target MoE model（Qwen2-57B-A14B，K=8，E=14）：N(32)≈14=满激活 → 验证 32×4=128 tokens 时仍仅激活 14 个 expert → T_T(32,4)≈T_T(32,1)（计算增量在 memory-bound 下近乎免费）→ speedup ≈ 2.29×。对比 baseline 的 B=1：N(1)≈8，验证时 N(4)≈13.5，多加载 5+ 个 expert → T_T(1,4)≈1.7×T_T(1,1)。
  关键设计决策：
  - 通过理论推导 N(t) 和 Texp(t;ρ) 精确刻画了 batch size B 与 sparsity ρ 如何共同决定 SD 有效加速窗口。
  - 提出 target efficiency 指标使研究者能独立评估系统因素对 SD speedup 的影响，补充了仅关注 acceptance rate 的不足。
  - 性能建模以 roofline model 为核心，引入可解释的拟合参数（bias=kernel launch overhead，k1/k3=roofline 强度，k2=单 expert 加载时间），使端到端加速透明可解释。
  - 对 private serving、latency-critical、memory-constrained 等实际场景的适用性分析，证明理论发现在实际部署中的价值。
  - 编译框架/Kernel调度/硬件架构/芯片设计：论文未明确说明。
