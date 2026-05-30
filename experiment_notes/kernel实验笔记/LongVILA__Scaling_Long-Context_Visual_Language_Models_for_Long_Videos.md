## LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现：MM-SP系统的核心是2D-Attention机制的kernel调度策略，包括：(1) 2D-Attention通信调度：构建N_head × N_ring二维通信mesh，intra-node使用All-to-All (A2A) 按attention head维度重分布QKV tensors并重新按sequence维度分区，inter-node使用Point-to-Point (P2P) 传输partitioned KV chunks。该设计充分利用intra-node NVLink高带宽(900 GB/s)和inter-node InfiniBand带宽(50 GB/s)的异构特性；(2) 负载均衡调度：RingAttention存在因果注意力三角形导致rank间计算不均衡（rank0首轮后idle，rank3持续计算），ZigZag-RingAttn通过序列重排序解决，2D-Attention在此基础上进一步通过A2A操作按head dim分布QKV，实现均衡计算；(3) 两阶段sharding调度：视觉编码阶段按图像数均分（避免视觉模态异构导致负载不均），LLM解码阶段按token数均分（dummy token padding确保均匀可分），仅在训练开始时执行一次重分布。

  实验比较：
  (a) Training throughput (Figure 9)：32 H100 GPUs上MM-SP vs ZigZag-RingAttn (2.1×-5.7×)，vs Megatron-LM CP (3.1×-4.3×)，vs Megatron CP+TP hybrid (1.1×-1.4×)，vs DeepSpeed-Ulysses (持平)。
  (b) Max sequence length scaling (Figure 9)：256 GPUs上MM-SP支持2M+ tokens。Ulysses受限于head数（GQA 8 KV heads/32 Q heads），2D-Attention可超越head数限制扩展。
  (c) 64 H100 GPUs (Table 8)：2D-Attention在578K序列16.9s/iter vs ZigZag-RingAttn 77.2s/iter，Megatron-LM CP/CP+TP在320K以上全部OOM。
  (d) FSDP vs Zero-3内存 (Table 7)：2D-Attention + FSDP在320K序列11.12s/iter，Zero-3 OOM。
  (e) Communication overlap副作用 (Table 2)：Ring-style SP的通信-计算重叠kernel占据SM资源，forward kernel慢4.2%-18.6%，backward慢0.5%-5.8%，证明计算-通信重叠设计在attention kernel上反而降低性能。
  (f) 两阶段sharding加速 (Table 5)：8 GPUs 1.12s/iter vs 1.20s/iter（7%加速）。
  (g) Convergence evaluation (Figure 12)：2D-Attention vs pure data parallelism训练loss曲线一致，证明MM-SP不影响训练质量。
  (h) Model complexity profiling (Table 10)：LLM Attention FLOPs随帧数二次增长（256 frames: 109.89 TFLOPs 1.5B / 256.40 TFLOPs 7B），是长视频最大计算瓶颈，验证MM-SP优化的必要性。

- 后端平台是什么，配置是什么。
  GPU: NVIDIA H100 80GB (NVLink 900 GB/s intra-node, InfiniBand 50 GB/s inter-node, 18×带宽差异)。节点间通过400 Gbps InfiniBand互联。扩展实验：NVIDIA A100 80GB节点（256 GPU验证最大序列长度）。推理：单节点8×H100。Profiling：单张A100 GPU, FP16, Flash-Attention2。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：基于HuggingFace Transformers的monkey-patched训练/推理系统。Attention后端使用Triton (Tillet et al., 2019) 实现2D-Attention kernel，配合Flash-Attention2 (Dao, 2024) 进行本地SDPA计算。FSDP (Zhao et al., 2023) 用于数据并行sharding。
  修改内容：(1) 实现2D-Attention通信调度：A2A + P2P混合，替代纯Ring P2P；(2) 实现两阶段sharding：视觉编码和LLM解码分别优化；(3) 实现推理模式SP：管理动态token/position encoding；(4) 使用Triton而非C++编写kernel（论文指出port到C++可获进一步加速）；(5) 集成Flash-Attention2作为本地注意力计算后端。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：github.com/NVlabs/VILA/tree/main/longvila

  评估原理：通过测量固定iteration数的训练时间计算throughput (iterations/second)，使用10 warmup iterations + 5 measurement iterations消除方差。逐步增加per-GPU sequence length从1K到10K直到OOM来测试最大序列长度。对8B模型(B=1)在32/64/256 GPU规模下评估。

  Kernel输入到性能输出全过程（以2D-Attention 4×2 mesh, seq_len=256K, 8 H100为例）：
  ```
  输入: local_tokens [B=1, seq_len/sp_size=32K, hidden=4096]
        sp_mesh: (A2A_group_size=4, P2P_group_size=2)
  
  Step 1: QKV Projection (local, no comm)
    Q,K,V = Linear(local_tokens)  → [1, 32K, num_heads=32, head_dim=128]
  
  Step 2: A2A Reshuffle (intra-node, NVLink 900 GB/s)
    # 4 GPUs within node, all-to-all by head dim
    # Input: Q sharded by sequence [1, 32K, 32, 128] per GPU
    # After A2A: Q sharded by head [1, 128K, 8, 128] per GPU
    Q_new = all_to_all(Q, scatter_dim=head, gather_dim=seq)
    K_new = all_to_all(K, scatter_dim=head, gather_dim=seq) 
    V_new = all_to_all(V, scatter_dim=head, gather_dim=seq)
    # Communication volume: 3 × 32K × 32 × 128 × 2 = ~24MB per GPU
    # Bandwidth: 900 GB/s → ~27μs for A2A
  
  Step 3: P2P KV Transfer (inter-node, InfiniBand 50 GB/s)
    # 2 GPUs across nodes, ring topology
    # Each sends K,V chunks to ring neighbor
    send(K_chunk[cur_rank], dst=(rank+1)%2, ring_group)
    recv(K_chunk[prev_rank], src=(rank-1)%2, ring_group)
    send(V_chunk[cur_rank], dst=(rank+1)%2, ring_group)
    recv(V_chunk[prev_rank], src=(rank-1)%2, ring_group)
    # Communication volume: 2 × 128K × 8 × 128 × 2 = ~2MB per GPU
    # Bandwidth: 50 GB/s → ~40μs for P2P
  
  Step 4: Local Attention (Tensor Cores, FlashAttention2)
    # K_all = concat(K_local, K_received_from_ring)
    # V_all = concat(V_local, V_received_from_ring)
    attn_output = FlashAttention2(Q_new, K_all, V_all, causal_mask=True)
    # Compute: O(S²×d) = (2×32K)² × 128 ≈ 0.26 TFLOPs per layer
    # Tensor Cores utilization: ~50-70% MFU
  
  Step 5: Reverse A2A (intra-node, NVLink)
    attn_output = all_to_all(attn_output, scatter_dim=seq, gather_dim=head)
    # Restore original head distribution: [1, 128K, 8, 128] → [1, 32K, 32, 128]
  
  输出: attn_output [1, 32K, 32, 128] → project to [1, 32K, 4096]
  
  性能对比（32K-256K序列）：

  | seq_len | ZigZag-Ring  | 2D-Attention | Speedup |
  |---------|-------------|--------------|---------|
  | 32K     | 2.06s       | 1.04s        | 2.0×    |
  | 64K     | 4.40s       | 1.08s        | 4.1×    |
  | 128K    | 8.63s       | 2.30s        | 3.8×    |
  | 256K    | 17.54s(OOM w/ Zero-3) | 7.04s(FSDP) | 2.5× |
  | 320K    | OOM(Zero-3) | 11.12s(FSDP) | N/A     |

  MM-SP的关键加速来源：
  - A2A利用NVLink 900GB/s全带宽 vs Ring P2P同样走NVLink但每个GPU仅与2个邻居通信
  - 2D-Attention将通信分两层：高频A2A走快通道(NVLink)，低频P2P走慢通道(InfiniBand)
  - Ring-Attention的P2P通信无法被计算完全掩盖，反而因占据SM资源拖慢attention kernel
