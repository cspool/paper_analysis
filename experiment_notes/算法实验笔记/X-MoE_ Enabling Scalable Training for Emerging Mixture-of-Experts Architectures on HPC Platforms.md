## X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：X-MoE 提出三项系统级算法优化以支持 expert-specialized MoE（DeepSeek 风格，fine-grained experts + large top-k routing）在非 NVIDIA HPC 平台上的大规模训练：
    (1) **PFT（Padding-Free Token buffers）+ 全 padding-free MoE pipeline**：设计稀疏数据结构 PFT（token_buffer x + ERI-arrays: token_ids, expert_ids, tokens_per_expert, combine_weights），消除传统 MoE dispatch/MLP/combine 各阶段的 zero-padding。PFT 仅存储有效路由 token，dispatch 使用 uneven alltoall 替代 even alltoall，通信量和激活内存随实际 token 数（非 capacity）线性增长。复杂度从 GShard 的 O(ckbsh)+O(ckb²s²) 降至 O(kbsh)。
    (2) **RBD（Redundancy-Bypassing Dispatch）**：分层两级 dispatch——Stage 0 选择 Pilot tokens（跨节点去重后的最小 token 集）和 Local replica（节点内重复的 token）；Stage 1 仅 Pilot tokens 通过跨节点 uneven alltoall + 节点内从 Pilot 重建 Local replica；Stage 2 Local replica 通过快速 intra-node alltoall 分发。消除 Dragonfly 拓扑下因 large top-k 导致的跨节点重复通信（实测冗余率可达 75.1%）。
    (3) **SSMB（Sequence-Sharded MoE Blocks）**：在 TP+EP 混合并行中，进入 MoE block 时将序列切分到 EP ranks（drop partial tokens），使 Adispatch 和 Acombine 激活内存按 TP group size 比例缩减。MoE block 结束后通过 all-gather 恢复完整序列。解决 expert-specialized MoE 中激活内存（尤其是 dispatch/combine 阶段）从模型参数转移的瓶颈。
  - 实验比较：(a) 可训练性与吞吐量：X-MoE vs DeepSpeed-MoE vs DeepSpeed-TED vs Tutel，在 Small（10.1B）/Medium（55.2B）/Large（201.4B）/Super（545.4B）四种 DeepSeek 风格模型配置上，256-1024 AMD MI250X GPU，对比训练吞吐量（TFLOPs）和 OOM 情况；(b) Weak scaling（16→256 GPU）和 Strong scaling（128→1024 GPU）；(c) MoE layer 时间分解：X-MoE vs DeepSpeed-MoE 的 gating/dispatch/alltoall/expert compute/combine 各阶段延迟；(d) 激活内存：X-MoE vs DeepSpeed-MoE vs Tutel 每 MoE layer 内存消耗；(e) SSMB vs activation checkpointing 的吞吐量对比；(f) Cross-platform 验证：8×NVIDIA A100 40GB 上 X-MoE vs DeepSpeed-MoE vs Tutel。

- 硬件平台是什么，配置是什么。
  - 主平台：Frontier 超级计算机（OLCF），每节点 4×AMD MI250X GPU（每 GPU 2 GCD，视为独立 GPU），GCD 间 Infinity Fabric 互联（50-200 GB/s 峰值），节点间 Slingshot 25 GB/s NIC（Dragonfly 拓扑），最多使用 128 节点（1024 MI250X GCD）。每 effective GPU 峰值 191.5 TFLOPs。
  - 跨平台验证：8×NVIDIA A100 40GB GPU。

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-MoE 风格的 expert-specialized MoE。四个配置：
    - Small: H=2048, HFFN=1408, 64 experts, top-k=6, 28 layers, 10.1B params (1.3B activated)
    - Medium: H=5120, HFFN=1536, 128 experts, top-k=6, 28 layers, 55.2B params (5.2B activated)
    - Large: H=7168, HFFN=2048, 256 experts, top-k=8, 28 layers, 201.4B params (11.5B activated)
    - Super: H=7168, HFFN=2560, 256 experts, top-k=8, 61 layers, 545.4B params (28.7B activated)
  - 数据集：论文未明确说明具体训练数据集名称，使用标准 LLM 预训练数据。Benchmark：训练吞吐量 (TFLOPs)、迭代时间、激活内存消耗、alltoall 延迟。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Supercomputing-System-AI-Lab/X-MoE，集成于 DeepSpeed 0.15.5。
  - PFT padding-free MoE pipeline 伪代码（对应 Listing 1）：
    ```
    # === Step 1: Gating ===
    # tokens: [S, H], S=sequence_length, H=model_dim
    logits = softmax(FFN(tokens), axis=-1)  # [S, E]
    combine_weights, top_experts = topk(logits, k)  # [S, K] each

    # === Step 2: PFT Construction ===
    # Input: top_experts [S, K], combine_weights [S, K], max_token_count
    flat_top_experts = flatten(top_experts)  # [S*K]
    flat_combine_weights = flatten(combine_weights)  # [S*K]
    sorted_indices = argsort(flat_combine_weights)
    sorted_top_experts = flat_top_experts[sorted_indices]
    # Token dropping: one_hot + cumsum + mask
    one_hot_enc = one_hot(sorted_top_experts, num_classes=E)  # [S*K, E]
    rank_in_expert = cumsum(one_hot_enc, axis=0)  # [S*K, E]
    weight_mask = rank_in_expert <= max_token_count
    # Filter retained tokens → ERI-arrays
    filtered_indices = sorted_indices[weight_mask]
    token_ids = token_ids[retained_token_ids]  # [B], B=retained tokens
    expert_ids = expert_ids[retained_token_ids]  # [B]
    combine_weights = combine_weights[retained_token_ids]  # [B]
    tokens_per_expert = histogram(expert_ids, bins=E)  # [E]

    # === Step 3: Padding-free Dispatch ===
    # Gather kernel: reorder tokens locally per expert routing
    dispatch_in = gather_kernel(gate_out, pft.token_ids, pft.expert_ids)  # [B, H]
    # Uneven alltoall: only valid tokens, no zero-padding
    pft.tokens_per_expert = alltoall(pft.tokens_per_expert)
    dispatch_out = alltoallv(dispatch_in, pft.tokens_per_expert)  # [Bexp, H]

    # === Step 4: Padding-free MLP (sequential GeMM) ===
    # For expert i (0..E_local-1), process tokens 
    #   dispatch_out[sum(tpi[:i]):sum(tpi[:i+1])] with expert_i weights
    inter_activ = sequential_gemm(pft.x, w1)  # [Bexp, HFFN]
    mlp_out = sequential_gemm(inter_activ, w2)  # [Bexp, H]

    # === Step 5: Padding-free Combine ===
    combine_in = alltoallv(pft.x, pft.tokens_per_expert)  # [B, H]
    combine_out = scatter_kernel(combine_in, pft.token_ids,
                                  pft.expert_ids, pft.combine_weights)  # [S, H]
    ```
  - RBD 分层 dispatch 流程：
    ```
    # Stage 0 (S0): Pilot Selection
    # 对每个 token 的 k 个 destination experts，按 destination node 分组
    # 每组随机选 1 个 pilot token，其余标记为 local replica
    # 构建 s1_mapping_indices: local replica → pilot token 的映射

    # Stage 1 (S1): Inter-Node Exchange (Pilot Only)
    pilot_tokens = gather_kernel(x, pilot_token_ids)  # 仅 pilot tokens
    pilot_tokens = alltoallv(pilot_tokens, ...)  # 跨节点 uneven alltoall
    # Local replica 在目标节点从对应 pilot token 重建:
    local_replica[i] = pilot_tokens[s1_mapping_indices[i]]

    # Stage 2 (S2): Intra-Node Exchange (Local Replica Only)
    local_replica = intra_node_alltoallv(local_replica, ...)  # 节点内快速 alltoall
    # Merge pilot + local replica + reorder by expert index
    ```
