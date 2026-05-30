## All-Gather (AG) Communication for Expert Distribution in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

All-Gather (AG) 是一种 NCCL 集合通信原语，在 MoE 训练中用于将 expert 参数从一个 GPU 分发到一组 GPU。与 All-to-All 不同，AG 的特点是每个 GPU 的发送数据被收集到通信组中所有 GPU 上（各 GPU 输出 = 所有 GPU 输入的全集拼接）。在 HybridEP 的混合 EP 方案中，AG 替代传统的跨 DC A2A 通信来传输 expert 参数：域内 GPU 通过 AG 收集彼此的压缩 expert 参数，使得每个 GPU 都拥有域内所有 expert 的完整副本，从而消除了域内跨 GPU 的 token 数据传输（因为所有 expert 都已在本地可用）。AG 的关键优势：(1) Expert 的可压缩性——expert 权重分布比 activation data 更集中、outlier 更少，可从 P_E 压缩 50× 至 P_E/50；(2) 异步潜力——expert 不依赖 token data 即可传输，AG 通信可与 pre-expert 计算（Attention）完全重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HybridEP 中 AG 用于域内 expert 分发的伪代码：

```
# AG for Expert Distribution in HybridEP (S_ED=4, 4 DCs in Expert Domain)
# 每个 GPU 持有 1 个压缩 expert (P_E/CR), CR=50

# Asyn-comm 阶段：AG 与 pre-expert computation 重叠
# GPU i 的 CUDA stream:
stream_comm:
    # 1. SREncode 结果已在 Send Queue (与上一 iteration optimizer.step 融合)
    compressed_expert = send_queue.pop()  # 压缩后: value-index 格式, P_E/CR
    
    # 2. NCCL All-Gather: 域内所有 GPU 收集彼此的压缩 expert
    #    输入: 每个 GPU 贡献 compressed_expert_i [P_E/CR]
    #    输出: 每个 GPU 获得所有 GPU 的压缩 expert [S_ED * P_E/CR]
    all_compressed = NCCL_AllGather(compressed_expert, group=domain_group)

stream_compute:
    # 同时执行 pre-expert computation (Attention + FFN 前向)
    attn_output = attention(local_tokens)  # 与 AG 完全重叠

# 同步后:
stream_comm:
    # 3. SRDecode: 恢复完整 expert = shared_expert + decompress(residual)
    for i in range(S_ED):
        expert_i = SRDecode(all_compressed[i], shared_expert)
        recv_queue.push(expert_i)  # 供 expert FFN 使用

# Expert Computation:
for expert in recv_queue:
    output += gate_weight * expert_ffn(expert, tokens)
```

AG 通信量分析: 未压缩时 $V^{AG} = P_E * (S_{ED} - 1)$，压缩后降至 $P_E/CR * (S_{ED} - 1)$。以 Mistral-Small P_E=4.7MB, S_ED=8, CR=50: V_AG ≈ 4.7/50 * 7 ≈ 0.66MB per GPU（vs 未压缩的 32.9MB）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NCCL 实现：`ncclAllGather(sendbuf, recvbuf, sendcount, datatype, comm, stream)`，HybridEP 调用 NCCL-2.10 实现。
- 与 NCCL All-to-All 的区别：AG 每个 GPU 的发送量小（仅自己的 expert）但接收量大（所有 GPU 的 expert 合集）；A2A 每个 GPU 的发送和接收量通常更平衡但需要更复杂的 send/recv 配对。在跨 DC 低带宽场景下，AG 的发送数据可以充分压缩（expert 残差的 Top-k），而 A2A 的 token data 无法同等程度压缩。
- 异步实现关键：利用 CUDA stream 将 AG 通信放在独立 stream 中，通过 CUDA event 同步。HybridEP 的 Asynchronous Communicator 管理 Send Queue（编码后 expert）和 Recv Queue（解码后 expert），实现 AG 通信与计算的完全异步重叠。
- AsyncEP (ZeRO-Prefill, 2026) 采用类似思路：用异步 weight AllGather 替换 per-layer activation AllToAll，weight streaming 与 prefill compute 完全重叠，在长序列 prefill 场景取得 1.35-1.37× 加速。
- SHARP (NCCL 2.27+) 支持 AllGather 的 In-Network 计算，可将 AG 延迟降低 2.5×（NVSwitch 系统），对 AG 的使用有显著利好。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
- IFMoE: An Inference Framework Design for Fine-grained MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Parallelism (EP) 是 MoE 模型训练的专用并行策略（Shazeer et al., 2017）。在 EP 中，MoE 层的多个专家被分配到不同的 GPU 设备上，每个 GPU 持有部分专家的参数。训练时，每个 GPU 上的 token 根据 gate 路由被 dispatch 到持有目标专家的 GPU，计算完成后结果 combine 回原 GPU。与 Data Parallelism (DP) 不同，EP 中各 GPU 持有不同的模型参数（不同专家），而非相同参数的副本。EP 的关键通信代价是 all-to-all (A2A)。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FOLDMOE 中的 EP 配置：

```
配置: EP=16 (16 GPUs, 每 GPU 1 expert)
并行组合: attention 层用 DP=2 + TP/SP=8, MoE 层用 EP=16

单 GPU 上的 MoE layer forward with EP:
while True:
    # 1. Gate 计算 (本地, 无通信)
    gate_scores = gate(local_tokens)      # [num_local_tokens, num_experts=16]
    routes = topk(gate_scores, k=1)        # 每个 token 路由到 1 个 expert

    # 2. A2A Dispatch (跨 GPU 通信)
    for each expert e:
        tokens_to_e = gather(routes == e)  # 收集路由到 expert e 的所有 token
        send(tokens_to_e, dst=gpu_of_expert[e])
    remote_tokens = recv(from all GPUs)    # 接收其他 GPU 发来需要本 GPU 处理的 token

    # 3. Expert Compute (本地, 无通信)
    output = expert(local_expert, remote_tokens, gate_scores)

    # 4. A2A Combine (跨 GPU 通信)
    for each origin GPU:
        send(output_for_tokens_from_origin, dst=origin)
    combined_output = recv(from all GPUs)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- EP 通常与 DP/TP 混合使用：DP 在节点间，TP+SP 在节点内 attention 层，EP 在所有 GPU 上的 MoE 层
- EP 通信量 = 2 × num_tokens × d_model × sizeof(dtype)（dispatch + combine 各一次）
- 扩展：每 GPU 可持有多个专家（n_expert_per_gpu > 1），以减少 A2A 通信但增加每 GPU 计算量
- FOLDMOE 每 GPU 1 expert，这是 EP 通信压力最大的配置，突出了 A2A 瓶颈
- FSMoE 在 EP 基础上结合 Expert-Sharding Parallelism (ESP)，当 GPU 数量超过 expert 数量时，每个 expert 被进一步分片到多个 GPU (ESP group)。EP 和 ESP 的组合引入了额外的 ESP-AllGather 和 ESP-ReduceScatter 通信操作，FSMoE 通过协同调度将这些额外通信与 AlltoAll 重叠，进一步提升训练效率。

FUSCO 通过融合数据变换和通信来优化 A2A 性能。FUSCO 将 A2A 通信建模为 structured segments（token 级别的逻辑单元），使用 Segment Descriptor（{addr, size} 对数组）捕获每个 segment 的源/目标内存布局。在 dComm 引擎中，A2A Dispatch 的发送端 GPU kernel 根据 descriptor 从非连续内存（expert-major layout）gather 数据到 NIC ring buffer，inline 完成 layout transformation；接收端直接 scatter 到 expert activation tensor 的最终位置。dComm 还实现了 Hierarchical Routing：对同一目的节点的多个 expert，sender 仅发送一份 token 拷贝给 forwarder GPU，forwarder 再经 intra-node NVLink 分发，消除跨节点重复传输。

- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

HAP 对 EP 在 MoE 推理中阶段差异的分析：prefill 阶段（长序列，batch×seqlen 大），EP 的 All-to-All 通信量低于 TP 的 AllReduce——EP 仅传输路由到对应 expert 的 token 数据而非全量输出。因此 EP 在通信瓶颈场景（PCIe 低带宽如 A6000/V100）下优于 TP。但 decode 阶段（单 token），EP 的负载不均衡问题突出——热门 expert 所在 GPU 繁忙而其他 GPU 空闲，TP 因权重均匀切分无此问题。HAP 利用这一阶段差异，允许 Expert 模块在 prefill 用 EP、decode 切换为 TP，通过动态策略切换实现取两者之长、避两者之短。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

HybridEP 揭示了 EP 在跨数据中心（cross-DC）场景下的根本性瓶颈和一种新的解法。在跨 DC 部署中，inter-DC 带宽极低（10Gbps Ethernet vs intra-DC PCIe 128Gbps），EP 的 A2A 通信可占训练总时间的 50%-90%（Figure 2b），且通信时间（数十 ms）远超计算时间（<1ms），导致传统的计算-通信重叠策略（FasterMoE, Tutel, SmartMoE）完全失效——通信根本无法被隐藏。HybridEP 通过将 EP 改造为 Hybrid Expert/Data Transmission 来结构化解决此问题：(1) 引入 AG 通信替代部分 A2A——利用 expert parameter 的高可压缩性（50× via SR compression）和异步传输潜力，将跨 DC 的低带宽通信从 token data 转向 expert weight；(2) Stream-Based Modeling 自动决定最优的 A2A/AG 混合比例 p；(3) Domain-Based Partition 将 p 映射到层级 GPU 拓扑（域内 AG，域间 A2A）；(4) 当 p=0（纯 AG）时，标准 EP 的跨 DC A2A token 传输完全消除。本质上，HybridEP 将 EP 泛化为一种更灵活的混合通信范式——标准 EP (p=1) 只是其特例。在 1000 DC 仿真中，HybridEP 相比 EP 最高 1.45× 加速（固定 domain size），相比 Tutel/FasterMoE/SmartMoE 最高 5.6× 加速（cross-DC 低带宽训练）。

HeterMoE 揭示了 EP 在异构 GPU 集群上的局限：EP 不区分 GPU 型号，将 attention 和 expert 统一分配，导致旧 GPU（V100）也被迫执行 attention（不支持 FlashAttention，64K 时仅 A40 的 27% attention 性能）。HeterMoE 提出 Zebra Parallelism 替代 EP——ZP group 内 attention 仅在新 GPU 复制，expert 仅分布在旧 GPU，通过 microbatch-level 跨 GPU 流水线实现 overlap。

FarSkip-Collective 的工作将 EP 概念扩展到推理侧（vLLM/SGLang），使用了不同于训练的 EP 实现方式。在 vLLM/SGLang 推理 EP 中，activation 在所有 rank 上复制，仅 expert 权重按 EP 分布，使用 all-reduce（而非 all-to-all）聚合结果。FarSkip 将此 all-reduce 异步化，利用架构修改后的依赖断裂点实现通信-计算重叠（all-reduce 重叠率 95.3-97.6%）。

Hecate 量化了 EP 的 **straggler effects**：imbalanced expert load 下，最重载 device 决定了整个 MoE layer 的计算延迟（其他 device 等待），同时该 device 的入站 All-to-All 通信量也最大。在 AWS V100 cluster 上评估，相比 balanced load 分布，imbalanced load 可使训练性能下降 5.18×。Hecate 的 FSSDP 通过 SparseAllGather/SparseReduceScatter 替代 EP 的静态 expert 分布，每 iteration 从零构建临时 placement，使 expert load 的 straggler 效应被稀疏物化机制消除。在 All-to-All 层面，Hecate 的拓扑感知 dispatching（优先 intra-node）使 A2A 通信时间相比 EP 减少 12.3×。
