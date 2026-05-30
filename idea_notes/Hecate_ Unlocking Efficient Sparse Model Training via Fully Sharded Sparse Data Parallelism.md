## Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

- baseline方法是什么？
  - **Expert Parallelism (EP)**：MoE layer 的 experts 均匀分布到多个 device，每个 device 持有若干完整 expert 的参数和 optimizer states。Token 通过 All-to-All 通信 dispatch 到持有对应 expert 的 device，计算完成后 All-to-All 收集回原 device。但由于 MoE gate 训练的随机性，expert loads 频繁波动和不平衡（图 3），导致最重载 device 成为通信和计算的 straggler，拖慢整个 MoE layer 的执行时间。评估显示 EP 在最坏 load imbalance 下性能下降达 5.18×。
  - **Expert Rearrangement Systems（FasterMoE, SmartMoE, FlexMoE）**：在 EP 基础上，通过动态调整 expert placement（重排/复制 expert）来减轻 straggler。但面临两个核心挑战：(C1) **Memory challenge**：更 balance 的 placement 需要更多内存来容纳 replica experts 及其 optimizer states，预留内存不足会限制 placement 优化空间（FlexMoE 实验中 4× 内存仅换 2.65× speedup）；(C2) **Timeliness challenge**：rearrangement 频率的 trade-off——频率高则 placement 更 timely 但通信开销大，频率低则 placement 过时；某一场景的最优频率无法泛化到其他场景（SmartMoE 实验中每 10 steps 相比每 25 steps，non-rearrangement iteration 快 2.9% 但整体慢 10.2%）。
  - 全栈执行例子（Baseline FlexMoE 训练 GPT-MoE-S on Cluster B, 32 A100 GPUs）：
    - **模型推理/训练算法层**：标准 MoE training loop。MoE gate 做 Top-2 token-to-expert assignment → expert FFN 计算（W_gate, W_up, W_down GEMMs）→ gate loss backward + expert backward。FlexMoE 按 token-to-expert 分配统计，启发式决策 expert replica 创建/删除 → AllReduce 同步 replica gradients。
    - **系统框架层**：PyTorch + Megatron-LM 训练框架。Expert parallelism 通过 All-to-All collective 实现 token dispatching。FlexMoE 的 rearrangement manager 在 iteration 间迁移 expert 参数+优化器状态（参数 6× 大小的量级，Adam optimizer mixed precision 下 optimizer states 至少 6× parameters），通过 P2P 通信进行 expert relocation/replication。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，NCCL collective communication backend，cuBLAS GEMM。
    - **kernel 调度层**：NCCL All-to-All 通信 kernel（token dispatching）+ cuBLAS GEMM kernel（expert FFN 计算）。FlexMoE 的 rearrangement 引入额外 NCCL P2P Send/Recv（expert 参数+优化器状态传输，在 critical path 上）。AllReduce（gradient sync of replicated experts）在 backward 结束时执行。通信量：对于 placement P'，每个 DP group D_i 做 AllReduce for expert e_i，总通信量 O(2λS)。
    - **硬件架构层**：A100-40G GPU × 32（4 nodes × 8 GPUs），NVSwitch 600 GB/s intra-node，400 Gbps NIC inter-node。V100-32G × 32（4 nodes × 8 GPUs），NVLink 300 GB/s intra-node，100 Gbps inter-node。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Hecate 方法**：提出 Fully Sharded Sparse Data Parallelism (FSSDP)，从全新角度解决 MoE 训练的 straggler 问题。核心思想受到 FSDP 的启发：**将 MoE layer 的参数和 optimizer states 完全分片到所有 device，每次 iteration 从 shards 中用稀疏通信原语 (SparseAllGather + SparseReduceScatter) 从零构建一个临时的 expert placement，而不需要在 iteration 间迁移 expert 状态**。
  - 三大设计对应解决 baseline 缺陷：
    1. **FSSDP 消除 rearrangement memory overhead（解决 C1）**：
       - Baseline 缺陷：rearrangement 系统需要预留内存来接收迁移的 expert 参数+优化器状态（6× 参数量），越 balance 的 placement 越消耗内存。
       - Hecate 方法：FSSDP 全局只保留一份 optimizer states（不再每个 replica 一份），sharding phase 将其均匀分布在所有 device 上。Materialization phase 仅物化 expert 参数（非 optimizer states），用完即释放。Heterogeneous sharding 跨层统一调度，进一步优化 underloaded expert 的 placement 减少 All-to-All congestion，同时保证内存均衡。Re-materialization 将物化参数的额外内存开销降低 90.2%，总内存仅比 EP 增加 11.6%。
    2. **Sparse collectives 消除 rearrangement timeliness trade-off（解决 C2）**：
       - Baseline 缺陷：rearrangement 在 critical path 上，频率越高 placement 越 timely 但通信开销越大，最优频率不可泛化。
       - Hecate 方法：FSSDP 将 "rearrangement" 的概念从 "在 iteration 间迁移 expert 状态" 变为 "在 iteration 内从 shards 物化临时 placement"。SparseAllGather 和 SparseReduceScatter 的通信量与同 placement 下 rearrangement 的 AllReduce 通信量等价（O(2λS)），但消除了额外的迁移通信。两个稀疏 collective 的通信与 Attention computation 重叠调度（forward 重叠 spAG，backward 重叠 spRS + spAG），使 sparse materialization 脱离 critical path。每 iteration 都能工作在当前最优 placement 下，不存在 timeliness trade-off。
    3. **拓扑感知的 materialization 和 dispatching（超越 baseline 的优化）**：
       - Hecate 的 sparse materialization (Algorithm 1) 在搜索 placement 时考虑 interconnect topology：overlap degree t 的计算使用 inter-node bandwidth（异构网络）或 uniform bandwidth（同构网络），优先 intra-node 通信。Token dispatching 同样优先 intra-node，减少 inter-node All-to-All congestion。这些拓扑感知设计使 Hecate 的 All-to-All 通信时间比 EP 减少 12.3×。
  - 全栈执行例子（Hecate 训练 GPT-MoE-S on Cluster B, 32 A100 GPUs）：
    - **模型推理/训练算法层**：FSSDP 替代 EP。Sharding phase：Heterogeneous sharding (Algorithm 2) 将 64 experts 跨 32 devices 分片，underloaded experts 优先按 node/device 负载均衡放置，overloaded experts 填充剩余 slots。Materialization phase (Algorithm 1)：基于滑动窗口 (w=5) 估计下轮 expert load → 在 overlap degree t 和 memory capacity m 约束下搜索 placement P' → Calibration stage（gate 输出后）用实际 token assignment 决定是否追加物化。Token dispatching 优先 intra-node。Backward 梯度通过 spRS reduce 到 MoE shard 所在 device → optimizer step 更新本地 shard。
    - **系统框架层**：PyTorch + Megatron-LM 框架。Hecate 的 Executor 驱动 FSSDP workflow（sharding → materialization → dispatching → compute → gradient reduce → optimizer）。Communicator 管理 NCCL sparse collectives 和 All-to-All 通信队列。Scheduler 生成 placement plan。Dispatcher 做拓扑感知 token 路由。
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：SparseAllGather = NCCL group calls (一组 Broadcast)，SparseReduceScatter = NCCL group calls (一组 Reduce)。Forward 中 spAG 与 Attention forward 重叠（约束：spAG 延迟 ≤ Attention fwd 延迟）。Backward 中 spRS (layer l) + spAG (layer l+1 re-materialize) 与 Attention backward 重叠（backward 耗时 ~2× forward，容纳两个 collective）。若启用 Hecate-RM：expert 参数 forward 后立即释放，backward 时重新 spAG 物化，增加 3.6× sparse collective 通信开销但仍优于 baseline 1.4×。
    - **硬件架构层**：A100-40G GPU × 32。Sparse collectives 利用 NVSwitch 600 GB/s intra-node 做高效 Broadcast/Reduce。Topology-aware scheduling 优先 intra-node 通信路径，减少 400 Gbps NIC inter-node 链路的拥塞。Hecate 在 Cluster A (V100, 100 Gbps inter-node) 上加速比更高（geo-mean 2.05× vs Cluster B 的 1.26× vs EP），因为低带宽环境的 All-to-All straggler 效应更强，Hecate 的拓扑感知优化收益更大。

- baseline方法是什么？
  - **Expert-Offloading 技术**：将 non-expert 权重 + 部分 "hot expert" 缓存于 GPU memory（expert cache），其余 expert offload 到 CPU memory 或 SSD，按需加载。当 cache miss 时，通过 PCIe/SSD 加载缺失 expert 并 evict 现有 expert。
  - Baseline 系统包括：
    1. **EdgeMoE**：对不同 expert 使用静态量化级别（基于特定数据集 profiling 确定 bit-width），缺乏跨环境灵活性。
    2. **AdapMoE**：激进跳过某些 expert 以减少加载开销，导致显著精度下降（特别是 small top-k 时，如 Mixtral-8x7B 的 k=2）。
    3. **MoE-Infinity**：按 expert activation ratio 优先级做 prefetching，但 prefetching 收益有限（expert 加载延迟 >> GPU 计算延迟）。
    4. **MoE-Offloading**：用当前层 gate input 预测下一层 expert（LRU 缓存策略），但预测带来的 overlap 收益同样受限于加载/计算比。
    5. 通用缓存策略（LFU、LRU）无法利用 mixed precision expert cache 中不同精度加载代价的差异。
  - 全栈执行例子（Baseline MoE-Infinity on RTX 4090，Mixtral-8x7B FP16）：
    - **模型推理算法层**：标准 MoE layer，router 计算 top-K=2 experts → 若 cache miss，从 CPU memory 加载完整 FP16 expert 权重（~10.5MB/expert for 7B hidden × 4096 FFN × 3 矩阵），传输时间 ~0.33ms (PCIe 4.0 32GB/s)。Expert FFN 计算：W_g/W_p/W_o GEMM，GPU 计算 ~3ms/layer。加载占 85.5% 总时间。
    - **系统框架层**：MoE-Infinity 基于 PyTorch + CUDA，expert 权重存储在 CPU memory (mmap)，LRU/LFU cache 管理 GPU expert cache。Prefetching 基于 expert activation 历史统计。
    - **编译框架层**：论文未明确说明。PyTorch eager mode，cuBLAS GEMM kernel。
    - **kernel 调度层**：标准 cuBLAS FP16 GEMM，expert loading 是单一大块连续内存拷贝 (cudaMemcpy)。所有 cache-miss expert 按相同 FP16 精度加载。
    - **硬件架构层**：RTX 4090 24GB + CPU 256GB，PCIe 4.0 ×16 (32GB/s)。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HOBBIT 方法**：混合精度 Expert Offloading 系统，通过三个层次的创新利用 MoE 计算的自然层次结构，核心洞察是 **"动态将不太关键的 cache-miss expert 替换为低精度版本可以显著减少 expert 加载延迟同时保持模型精度"**。
  - 三大设计对应解决 baseline 缺陷：
    1. **Token-level Dynamic Expert Loading（解决静态/激进量化/跳过问题）**：
       - Baseline 缺陷：EdgeMoE 的静态量化依赖特定数据集 profiling，不灵活；AdapMoE 的激进跳过导致精度显著下降。
       - HOBBIT 方法：用 ||G(x)|| 作为 expert 重要性的动态代理（与 ||G(x)E(x)|| Pearson r=0.99），计算 unimportance degree score，双阈值灵活决策精度。避免了静态 profiling，运行时自适应输入；用低精度替换替代跳过，精度下降 <1%。
    2. **Layer-level Adaptive Expert Prefetching（解决预取收益低问题）**：
       - Baseline 缺陷：prefetching 因 expert 加载时间 >> GPU 计算时间而收益有限，错误预测惩罚严重（无法中断 cudaMemcpy）。
       - HOBBIT 方法：利用层间 gate input 高余弦相似度（相邻层 top-1 准确率平均 96%），Stacking Computer 一次性批量计算所有后续层 gating。关键创新：用低精度预取替代高精度预取，即使预测错误，低精度 expert 的错误加载惩罚仅为高精度的 1/4，使预取在任何精度下都产生正向收益。
    3. **Sequence-level Multidimensional Expert Caching（解决低效缓存管理问题）**：
       - Baseline 缺陷：LRU/LFU 等通用策略不考虑 mixed precision 特性（高精度 miss 代价 4× 低精度 miss）。
       - HOBBIT 方法：提出 LHU (Least High Precision Frequently Used) 策略追踪高精度使用频次 H_t，与 LRU + LFU + FLD 四策略加权组合。高/低精度 cache 分离管理，按加权优先级公式 evict。序列级 LFU（同一 sequence 内统计）相比模型级 LFU 提升 4.5% hit ratio。
  - 全栈执行例子（HOBBIT on RTX 4090，Mixtral-8x7B FP16+INT4）：
    - **模型推理算法层**：每 token 进入 MoE layer → Router 计算 top-2 gate weights → ||G(x)|| 归一化后计算 s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}|| → e_0 得分 0（高精度）→ e_1 得分 = ||G(x)_{e_0}||，若 ≤T1 为高精度加载，≤T2 为 INT4 加载（4× 更小），>T2 跳过。INT4 expert FFN 计算使用量化 GEMM。精度下降 GSM8K 准确率最大仅从 0.52→0.51（FP16→FP16+INT4）。
    - **系统框架层**：基于 Llama.cpp（8,000 行 C++/C 修改）。权重分布：所有 non-expert + 多精度 expert cache 驻留 GPU memory。主线程 GPU 计算 + scheduler 线程异步加载。Dynamic Expert Loader 通过 read() 系统调用从 CPU memory 加载对应精度 expert。Adaptive Expert Predictor Stacking Computer 一次性矩阵乘预测后续层 expert。Multidimensional Cache Manager Policy Performer 维护 LRU/LFU/LHU/FLD 记录，加权公式决定 eviction。
    - **编译框架层**：论文未明确说明。Llama.cpp 原生 CPU/GPU 混合编译，使用 CUDA/OpenCL 后端。
    - **kernel 调度层**：除标准高精度 GEMM 外，低精度 (INT4/INT2) expert 使用对应的量化矩阵乘 kernel。expert loading 从单一 FP16 大块拷贝变为多精度分块异步加载（FP16 expert ~10.5MB vs INT4 expert ~2.6MB）。由于 67%/30%/3% 的精度分布，平均加载量大幅减少。
    - **硬件架构层**：与 baseline 相同（RTX 4090 24GB + CPU 256GB，PCIe 4.0 32GB/s）。关键差异在于 PCIe 传输量：baseline 加载 2 个 FP16 expert（~21MB），HOBBIT 平均加载 ~12.4MB（1.0×FP16 + 0.3×INT4 + 0.03×skip），传输时间减少 ~41%。
    - **关键结果对比**：
      - vs MoE-Infinity on RTX 4090：decoding speedup 2.30× (Mixtral) / 3.92× (Phi-MoE)，prefill latency 降低 14%/29%。
      - vs MoE-Offloading on RTX 4090：decoding speedup 3.21× (Mixtral) / 3.29× (Phi-MoE)，prefill latency 降低 51%/54%。
      - vs Llama.cpp on Jetson Orin：decoding speedup 13.0× (Mixtral) / 18.9× (Phi-MoE)。
      - 精度：GSM8K 和 TruthfulQA 上所有配置下 accuracy 下降 ≤1%。
