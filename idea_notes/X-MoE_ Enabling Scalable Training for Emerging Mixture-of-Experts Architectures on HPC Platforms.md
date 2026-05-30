## X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

- baseline方法是什么？
  Baseline 是**现有 MoE 训练系统（DeepSpeed-MoE、DeepSpeed-TED、Tutel）在非 NVIDIA HPC 平台上训练 expert-specialized MoE（DeepSeek 风格）**。具体痛点：
  1. **CUDA 强依赖**：DeepSpeed-MoE 和 Tutel 的 MoE kernel 实现深度绑定 CUDA，无法高效移植到 AMD ROCm 平台。在 AMD MI250X 上性能 <10 TFLOPs（<10% 峰值），Megablocks 高度集成 Megatron-LM 无法在 AMD 上运行。
  2. **Zero-padding 内存和通信膨胀**：现有 MoE 框架（GShard、DeepSpeed-MoE、Fairseq）使用固定 expert capacity C 的 batched matmul pipeline。dispatch mask [S, E, C] + expert buffers [E, C, H] 大量 zero-padding，在 expert-specialized MoE（数百 fine-grained experts + large top-k）中 dispatch/combine 激活消耗 >70% 总激活内存，且 zero-padding 随 alltoall 传输浪费通信带宽。
  3. **跨节点通信冗余**：Large top-k 路由（如 k=8）使同一 token 被发送到多个跨节点 expert，在 Dragonfly 等层次化网络拓扑上产生大量重复跨节点传输（冗余率可达 75.1%），现有系统不感知网络拓扑。
  4. **激活内存瓶颈转移**：Expert-specialized MoE 中 dispatch/combine 激活（Adispatch, Acombine）随 fine-grained factor m 线性增长，而中间 FFN 激活保持不变。现有 TP+EP 混合并行在进入 EP 时仍复制全序列激活，无法缓解新瓶颈。

  全栈执行例子（以 DeepSpeed-MoE 在 Frontier AMD MI250X 上训练 DeepSeek 风格 201B MoE 为例，256 GPU，EP=64）：
  - **模型训练算法层**：DeepSeek 风格 expert-specialized MoE（256 experts, top-k=8, H=7168, HFFN=2048），GShard 式 gating + expert capacity C=1.25 × avg_tokens_per_expert，token dropping 策略。训练使用 ZeRO-1 DP + EP。
  - **系统框架层**：DeepSpeed-MoE v0.15.5，dispatch mask [S, 256, C] + expert buffers [256, C, 7168] with zero-padding，einsum + batched matmul pipeline，even alltoall（含 padding token 通信）。结果：OOM，无法训练。
  - **编译框架层**：论文未明确说明。CUDA/ROCm 编译，无跨平台编译优化。
  - **kernel调度层**：PyTorch einsum dispatch + batched matmul（含大量 zero-padding 计算），CUDA kernel（无法在 AMD 上高效运行，fallback 到慢速 PyTorch 实现）。Redundancy rate 54.8%（EP=32 时），跨节点重复传输。
  - **硬件架构层**：Frontier 超级计算机，AMD MI250X GPU（Infinity Fabric intra-node 200 GB/s, Slingshot inter-node 25 GB/s, Dragonfly 拓扑）。Alltoall 跨节点延迟 >10× intra-node，且随 scale 增加出现 outlier（>500ms per collective）。

  Baseline 的核心缺陷：(a) **跨平台可移植性差**——CUDA 绑定无法在 AMD/ROCm 上高效运行，需 costly 的 ROCm kernel 重写；(b) **Zero-padding 导致内存 OOM**——dispatch/combine 阶段 padded buffer 消耗 >70% 激活内存，限制可训练的模型规模；(c) **通信效率低**——even alltoall 传输 padded data + large top-k 导致跨节点 token 重复传输，在层次化网络上带宽利用极差；(d) **现有并行策略不适用**——TP+EP 不减少 Adispatch/Acombine 激活内存，ZeRO-DP 也不减少激活。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **X-MoE，一个跨平台 MoE 训练系统，通过 padding-free sparse pipeline、redundancy-bypassing dispatch 和 sequence-sharded hybrid parallelism 三大技术系统性解决 expert-specialized MoE 在非 NVIDIA HPC 平台上的训练瓶颈**。

  1. **PFT + Padding-Free MoE Pipeline（解决 Zero-padding 内存/通信膨胀 + 跨平台可移植性）**：
     - 设计稀疏 PFT 数据结构（token_buffer x + ERI-arrays），仅存储有效路由 token，消除 dispatch/MLP/combine 全流程 zero-padding
     - 用 uneven alltoall 替代 even alltoall，通信量随实际 token 数线性增长
     - Triton 实现 gather/scatter/sequential GeMM kernel，硬件无关（AMD ROCm + NVIDIA CUDA 均支持），无需 per-platform kernel 重写
     - 激活内存从 GShard 的 O(ckbsh)+O(ckb²s²) 降至 O(kbsh)

  2. **RBD（解决跨节点通信冗余）**：
     - 分层两级 dispatch：Pilot tokens（去重后最小跨节点 token 集）+ Local replica（节点内重复 token）
     - 仅 Pilot tokens 走跨节点 alltoall（低带宽 Slingshot 25 GB/s），Local replica 在目标节点从 Pilot 重建后走节点内 alltoall（高带宽 Infinity Fabric 200 GB/s）
     - 跨节点通信减少 52.5%（实测），总体 dispatch 加速 1.55×

  3. **SSMB（解决激活内存瓶颈转移）**：
     - 在 TP→EP 转换时，将输入序列切分到 EP ranks（drop partial tokens），每个 EP rank 仅保留 1/G 序列片段（G=TP group size）
     - Adispatch 和 Acombine 内存减少 G×
     - MoE block 结束后 all-gather 恢复完整序列，保持与下游 TP block 兼容
     - 相比 activation checkpointing：无重计算开销 + 无额外 alltoall（checkpointing 需 6 次 alltoall/layer，SSMB 仅 4 次）

  全栈执行例子（以 X-MoE 在 Frontier AMD MI250X 上训练 DeepSeek 风格 545B Super MoE 为例，1024 GPU，EP=256，TP=1-2）：
  - **模型训练算法层**：DeepSeek 风格 expert-specialized MoE（256 experts, top-k=8, H=7168, HFFN=2560, 61 layers, 545.4B params）。PFT padding-free pipeline 消除 zero-padding 开销。RBD 减少跨节点通信冗余。SSMB 切分 MoE block 序列减少激活内存。
  - **系统框架层**：X-MoE 集成于 DeepSpeed 0.15.5。PFT construction → uneven alltoall dispatch（仅有效 token）→ sequential GeMM（per-expert 无 padding）→ uneven alltoall combine。RBD: Stage 0 pilot selection → S1 inter-node uneven alltoall (pilot only) → S1 local replica reconstruction → S2 intra-node uneven alltoall (replica only) → merge。SSMB: TP block → drop partial tokens → EP MoE block (PFT+RBD) → all-gather → next TP block。
  - **编译框架层**：论文未明确说明。Triton 作为跨平台 kernel 编译器（Triton IR → AMD ROCm / NVIDIA CUDA PTX）。
  - **kernel调度层**：Triton gather kernel（B thread-blocks, 256 threads/block, coalesced read along H dim）→ uneven alltoallv (RCCL + libfabric, 仅实际 token) → sequential GeMM (rocBLAS, 每 expert 独立 launch, 无 padding 计算) → Triton scatter kernel（coalesced write along H dim）→ uneven alltoallv combine。RBD 模式：Pilot token gather kernel → inter-node alltoallv (Slingshot 25GB/s) → s1_mapping_indices-based local replica reconstruction → intra-node alltoallv (Infinity Fabric 200GB/s)。
  - **硬件架构层**：Frontier 超级计算机，1024 AMD MI250X GCD（128 nodes）。Dragonfly 拓扑：同一 rack ≤256 GPU（低延迟），256+ GPU 跨 rack 通信 alltoall 延迟 >10× 升高 + 频率高发 outlier (>500ms)。X-MoE 通过 RBD 最大化 intra-node 通信利用 + 限制 EP=256 避免跨 rack 延迟剧增。总计 10.44 PetaFLOPs 聚合吞吐量。

  **Baseline 缺陷 → 方法映射表**：
  | Baseline 缺陷 | 论文方法 |
  |---|---|
  | CUDA 绑定，无法在 AMD 上高效运行 | Triton 跨平台 kernel（gather/scatter/sequential GeMM），ROCm/CUDA 均支持 |
  | Zero-padding 导致 >70% 激活内存浪费 + 通信膨胀 | PFT 稀疏数据格式 + 全 padding-free pipeline（uneven alltoall + sequential GeMM） |
  | Large top-k 产生大量跨节点重复 token 传输 | RBD：Pilot token + Local replica 两级 dispatch，跨节点通信减少 52.5% |
  | TP+EP 并行无法减少 Adispatch/Acombine 激活内存 | SSMB：MoE block 内序列切分，激活内存减少 G× |
  | Even alltoall 随 scale 增大通信开销剧增 | PFT uneven alltoall + RBD hierarchical dispatch |
  | Activation checkpointing 需额外 alltoall + 重计算 | SSMB 无额外通信 + 无重计算，吞吐量更高 |
