## HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

- 属于算法pipeline的实现是什么？实验比较什么？
  - HierMoE 提出两种拓扑感知的算法优化来加速 MoE 训练中的 AlltoAll 通信：
    1. **Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll)**：利用 GPU 集群的分层拓扑结构（如 4 层：Inter-Node/InfiniBand → Inter-QPI → Inter-NVLink → Intra-NVLink），在不同层次维度上对 token 进行去重（deduplication），消除因多个 expert 位于同一 GPU/group 而导致的 token 重复传输。通过性能模型公式选择最优维度 d*，在高层（如 Inter-Node）减少低带宽链路上的通信量，将更多通信转移到高带宽的 Intra-node 链路。
    2. **Hierarchical Expert Swap (HierD-ES)**：在 HierD-AlltoAll 基础上，通过交换 expert 在 GPU 间的位置来平衡各 hierarchical group 的通信负载。计算交换任意两个 expert 后的估计通信时间矩阵 Q_d*，选择使通信时间最小化的 expert pair 进行交换。使用 smooth-max 函数平滑 Q_d 的梯度，提升优化稳定性。
  - 实验比较：
    - Baselines：Megatron-LM（标准 AlltoAll）、Tutel-2DH（二维分层 AlltoAll）、SmartMoE（expert placement 优化）
    - 消融：HD2-MoE（仅 2D 去重）、HD2-MoE-Smart（2D 去重+SmartMoE swap）、HD-MoE（HierD-AlltoAll 无 HierD-ES）、HierMoE（完整方案）
    - 评估指标：端到端训练加速比、AlltoAll 通信加速比
    - Ablation：不同 K（top-K experts）、E（expert 数）、G（GPU 数）下的加速比；不同层级维度的效果；不同 max 函数类型；不同 expert swap 更新频率

- 硬件平台是什么，配置是什么。
  - **32-GPU 集群**：4 节点 × 8 NVIDIA RTX A6000-48G GPU
  - 每节点配置：Dual Intel Xeon Platinum 8358 @ 2.60GHz，512GB DDR4，8× A6000-48G @ 1.46GHz
  - 互联：NVLink 112.5GB/s (4× link)，PCIe 4.0 (x16)，Mellanox MT28908 @ 200Gb/s InfiniBand
  - 软件环境：Ubuntu 20.04，CUDA 12.1，PyTorch 2.1.2，NCCL 2.18.5

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - DeepSeek-V3（hidden/model dim 减半至原始的 1/2，6 layers，EP degree=32）
    - Qwen3-30B-A3B（32 layers，EP degree=32）
  - **训练配置**：micro batch size=1，sequence length=1024
  - **数据集**：论文未明确说明具体训练数据集名称
  - **指标**：AlltoAll 通信时间加速比、端到端训练迭代时间加速比

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - HierMoE 本身未公开独立开源仓库，基于 Megatron-LM (https://github.com/NVIDIA/Megatron-LM/) 实现。NCCL 通信性能参数通过 nccl-tests (https://github.com/NVIDIA/nccl-tests) 采集拟合。
  - HierD-AlltoAll 算法 pipeline 伪代码（单 MoE layer 的 AlltoAll Dispatch 流程）：

```
=== 初始化阶段（集群启动时执行一次） ===
Input: 集群拓扑 D（4 层: Node/QPI/NVLink/Intra-GPU），GPU 数 G
Output: U[0..D-1]（各层 expert group 数），α/β 参数（AlltoAll 性能模型）

1. 通过 nccl-tests 测量 7 种 AlltoAll 通信的 α, β:
   - 标准 AlltoAll (HD1), Inter-Node (HD2-Inter1), Intra-Node (HD2-Intra1)
   - Inter-QPI (HD3-Inter2), Intra-QPI (HD3-Intra2)
   - Inter-NVLink (HD4-Inter3), Intra-NVLink (HD4-Intra3)
2. 使用最小二乘法拟合线性模型: t = α + n · β
3. U ← [1, 4, 8, 16, 32]  // 对应每层的 expert group 数
   例: Inter-Node AlltoAll 按 4 个 node 分成 4 组 (U[1]=4)
       Inter-QPI 每个 node 内再分 2 组 (U[2]=8)
       Inter-NVLink 再分 2 组 (U[3]=16)
       Intra-NVLink 最终 32 GPU (U[4]=G=32)

=== 每 iteration 的 HierD-AlltoAll Dispatch 流程 ===
Input: 路由结果 mask I_route ∈ R^{T×E} (T 为 token 数, E 为 expert 数)
       M (embedding 维度), G, E, D, U[], α/β 参数
Output: 最优维度 d*, 完成 token dispatch

Step 1: 计算 HD1-AlltoAll 的通信时间 t1
  m ← E/G                                    // 每 GPU 的 expert 数
  I_route^(1,G)[i,j] ← OR over j1 in [(j-1)m+1, j·m] of I_route[i,j1]
  p[j] ← sum_i I(I_route^(1,G)[i,j])          // 每 expert group 的去重 token 数
  n_a2a ← G · max(p) · M · v                  // 通信量 (v=字节/维度, FP16 下 v=2)
  t1 ← α_a2a + n_a2a · β_a2a

Step 2: 对 d = 2..D 计算 HDd-AlltoAll 通信时间 td
  for k = 1 to D-1:                            // 遍历各层级的 Inter-level
    m ← E/U[k]
    // 将 routing mask 聚合到 U[k] 个 expert group (去重)
    I_route^(k,U[k])[i,j] ← OR over j1 in [(j-1)m+1, j·m] of I_route^(k,E)[i,j1]
    p_a2a^(k,U[k])[j] ← sum_i I(I_route^(k,U[k])[i,j])
    n_a2a^Inter(k) ← (U[k]/U[k-1]) · max(p_a2a^(k,U[k])) · M · v
    // 更新 routing mask 以反映 Inter-level-k 通信后的 expert 分布
    I_route^(k+1,E) ← process(I_route^(k,E))
    p_a2a^(k+1,G)[j] ← sum_i I(I_route^(k+1,E)[i,j])
  // Intra-level-(d-1) 通信量
  n_a2a^Intra(d-1) ← (G/U[d-1]) · max(p_a2a^(d,G)) · M · v
  // 总时间 = Σ(Inter-level 各层) + Intra-level
  td ← Σ_{i=1}^{d-1} (n_a2a^Inter(i) · β_a2a^Inter(i) + α_a2a^Inter(i))
       + n_a2a^Intra(d-1) · β_a2a^Intra(d-1) + α_a2a^Intra(d-1)

Step 3: 选择最优维度 d*
  d* ← argmin_{1≤d≤D} td
  复杂度: O(D·T·K)

=== HierD-ES Expert Swap (每 iteration 可选执行) ===
Input: d*, routing mask, 当前 expert-to-GPU placement
Output: 交换的 expert pair (r*, c*)

1. 初始化 Z ∈ R^{E×E×U[d*]} 和 Z_intra ∈ R^{E×E×G}
   // Z[r,c,k]: 交换 expert r 和 c 后，第 k 个 expert group 的去重 token 数
2. for each token t (选中的 K 个 experts):
     for each expert pair (A, B) where A 被选中, B 未被选中:
       - Case 1/2: B 所在 group 无其他选中 expert → 该 group 计数+1
       - Case 2/4: A 是 group 内唯一选中 expert → A 原 group 计数-1
       - Case 1/3: A 所在 group 有 ≥2 选中 expert → A 原 group 不变
       - Case 3/4: B 所在 group 有选中 expert → B 所在 group 不变
   // 通过增量更新降低复杂度从 O(D·T·K·E²) 到 O(D·T·K·E)
3. 基于公式计算交换每对 expert 后的通信时间 Q_d*[r,c]
4. 使用 smooth-max (γ=10) 平滑: smooth_max(x,γ)=max(x)·(Σ_i (x[i]/max(x))^γ)^(1/γ)
5. (r*, c*) ← argmin Q_d*[r,c]
6. 交换 expert r* 和 c* 在 GPU 间的位置 (约 1% end-to-end 时间开销)
7. 将 Z 和 Z_intra 重置为无交换状态用于下一轮
```

- 去重效果量化：当 R=4（expert 组数）, K=8 时，重复率达 55%（表 II），HierD-AlltoAll 可消除这些重复。高 K（如 DeepSeek-V3 的 K=8）+ 低组数 → 高重复率 → 去重收益更大。

- 属于算法pipeline的实现是什么？实验比较什么？
  - Hecate 提出 **Fully Sharded Sparse Data Parallelism (FSSDP)**，一种全新的 MoE 训练范式，核心算法组件：
    1. **FSSDP Sharding Phase**：将每个 MoE layer 的 parameters 和 optimizer states 划分为 |𝒟| 个不相交的 MoE shards，每个 shard 包含一组 expert 的完整参数+优化器状态，全局只保留一份 optimizer states 副本，实现最小且均衡的内存占用。
    2. **SparseAllGather (spAG)**：用于每 iteration 从 MoE shards 中稀疏物化 (sparsely materialize) expert placement。形式化定义为 spAG(𝒫₀, 𝒫₁)，其中 𝒫₀ 为 surjective 的前置条件（每个 chunk 唯一归属于某 device），𝒫₁ 为 𝒫₀ 的超集。通信量上界 O(λS)，λ = |Ĉ|/|C| 为稀疏度，λ << 1 时远小于 FSDP 中 AllGather 的 O(S)。
    3. **SparseReduceScatter (spRS)**：用于将物化专家的梯度 reduce 回对应 MoE shard 所在 device。定义为 spRS(𝒫₀, 𝒫₁)，𝒫₁ ⊆ 𝒫₀ 且 surjective。每个 spAG(𝒫, 𝒫') 与对称的 spRS(𝒫', 𝒫) 配对。
    4. **Heterogeneous Sharding (Algorithm 2)**：跨所有 MoE layer 统一调度 sharding，允许每个 MoE shard 包含任意数量 expert（0 到 |ℰ|），同时保证跨 device 内存均衡。先放置 underloaded expert（负载变化慢），再填充 overloaded expert。re-sharding 低频触发（每 100 iterations）。
    5. **Sparse Materialization (Algorithm 1)**：拓扑感知的启发式搜索算法，在 overlap degree t 和 memory capacity m 两个约束下，搜索近似最优的 expert placement。t ≤ m 时，将 top-t overloaded expert 物化到所有 device；否则按负载比例分配 replica slots，优先 intra-node 通信。
    6. **Re-materialization**：物化的 expert 参数在用后立即释放，后续 backward 需要时再次通过 spAG 物化，将参数内存额外开销降低 90.2%。
  - 实验比较：baseline 包括 EP (原生 Expert Parallelism)、FasterMoE、SmartMoE、FlexMoE。评估 GPT-MoE-S (1.84B)、GPT-MoE-L (7.36B)、BERT-MoE (3.27B)、BERT-MoE-Deep (6.54B) 四种 MoE 模型在不同 GPU 规模（16/32 GPU）下的端到端训练加速比。消融实验分别验证 heterogeneous sharding 和 sparse materialization 各组件的贡献（图 15），以及 re-materialization 的 memory/performance trade-off（图 14）。layer-wise speedup 分析（图 11）和 critical path 时间分解（图 12）。

- 硬件平台是什么，配置是什么。
  - **Cluster A**：4× AWS p3dn.24xlarge nodes，每 node 8× NVIDIA V100-32G GPU（NVLink 300 GB/s），node 间 100 Gbps 网络。
  - **Cluster B**：4× AWS p4d.24xlarge nodes，每 node 8× NVIDIA A100-40G GPU（NVSwitch 600 GB/s），node 间 400 Gbps 网络。
  - 总规模：每个 cluster 32 GPUs（4 nodes × 8 GPUs），16 GPU 实验使用 2 nodes。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：GPT-3 和 BERT 的 MoE 变体。FFN 替换为 MoE layer（expert 仍为 FFN，d_ffn = 2 × d_model），GShard Top-2 gating。
    - GPT-MoE-S：d_model=768, SeqLen=2048, 12 layers, 64 experts, 1.84B params
    - GPT-MoE-L：d_model=1536, SeqLen=2048, 12 layers, 64 experts, 7.36B params
    - BERT-MoE：d_model=1024, SeqLen=512, 12 layers, 64 experts, 3.27B params
    - BERT-MoE-Deep：d_model=1024, SeqLen=512, 24 layers, 64 experts, 6.54B params
  - **数据集**：论文未明确说明具体数据集；训练框架使用 Megatron-LM，采用 weak scaling 方式（16 GPU 用 32 experts，32 GPU 用 64 experts）。
  - **指标**：端到端训练加速比（vs EP）、layer-wise 加速比、peak memory usage（optimizer states/gradients/parameters 分项）、critical path 时间分解。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未公开 Hecate 完整代码。基于 PyTorch + NCCL 实现，使用 Megatron-LM 作为训练框架。
  - FSSDP 算法 pipeline 伪代码（单 MoE layer l 在一个 iteration 中的执行流程）：

```
=== SHARDING PHASE (iterator 间低频执行) ===
Input: F^g (all MoE layers 的 expert load 分布), t (overlap degree)
Output: P^g = {P_0, P_1, ..., P_L} (各层 sharding plan)

1. J ← 各层 top-t overloaded experts
2. J' ← E^g - J  // underloaded experts
3. slots_per_device ← |E^g| / |D|  // 每 device 可用 slots
4. for each layer l in sortByMaxLoadDescending(L):
5.     for each expert e in J'_l (sorted by load descending):
6.         n ← least-loaded node (优先剩余 slots 少的)
7.         d ← least-loaded device on node n
8.         P_l ← P_l ∪ {(d, e)};  S_d ← S_d - 1
9. 将 J 中剩余 experts 任意分配到剩余 slots

=== MATERIALIZATION PHASE (每 iteration 执行) ===
Input: P (sharded placement), F (estimated expert loads),
       t (overlap degree), m (memory capacity)
Output: P' (materialization plan)

1. t ← min(t, |E|), m ← min(m, t)
2. P' ← P
3. if t ≤ m:
4.     E^topT ← Top t experts by load F
5.     P' ← P' ∪ (D × E^topT)  // 物化到所有 device
6. else:
7.     totSlots ← |D| · m
8.     for each e in sortByLoadDescending(E^topT):
9.         n ← assignSlotsByLoad(e, totSlots, F)
10.        P^e ← 在 nodes/devices 间分配 n 个 replica
               (优先有空闲 slots 的 node)
11.        P' ← P' ∪ P^e

=== FORWARD PASS of MoE layer l ===
1. // 通信与前一 Attention 计算重叠
2. P_l' ← Scheduler(P_l, F_l, t, m)  // Algorithm 1
3. spAG(P_l, P_l')  // SparseAllGather: 物化 expert 参数
4. // 可选 Calibration: 用 MoE gate 实际输出重新运行 Algorithm 1
5. Token dispatching (topology-aware All-to-All):
   - 优先 intra-node 通信
   - 同 expert 多 replica: 均匀分配 tokens
6. Expert FFN computation on materialized parameters
7. Release materialized parameters (若启用 re-materialization)

=== BACKWARD PASS of MoE layer l ===
1. spAG(P_l, P_l')  // re-materialize expert 参数
2. Expert backward computation
3. spRS(P_l', P_l)  // SparseReduceScatter: reduce gradients to source
4. Release materialized parameters

=== OPTIMIZER STEP ===
1. 各 device 在其 MoE shards 上用同步后的 gradients 更新
   optimizer states 和 model parameters
```

  - 关键张量计算：设 expert e_i 的参数为 W_i ∈ R^{d_model × d_ffn}（FFN 三层），SparseAllGather 从持有 e_i 的 source device d_src 以 Broadcast 方式将 W_i 发送到需要 e_i 的 target devices，通信量为 |W_i| × |target_devices|。spRS 逆过程，将各 device 上 e_i 的梯度 reduce（求和）回 d_src。整体通信量上界 O(2λS)，与同 placement 下 rearrangement 系统 AllReduce 通信量等价。
