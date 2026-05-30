## Multi-Modal Sequence Parallelism (MM-SP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MM-SP（Multi-Modal Sequence Parallelism）是LongVILA提出的针对视觉语言模型（VLM）的序列并行系统，通过在训练和推理阶段将单个长序列分布到多个GPU上进行并行计算，解决长视频VLMs训练的显存瓶颈。核心包括三个组件：(1) 两阶段Sharding——Stage1按图像数均匀分布帧进行视觉编码（避免视觉模态负载不均），Stage2按token数均匀切分进行LLM解码；(2) 2D-Attention通信机制——构建N_head × N_ring的二维通信mesh，intra-node使用All-to-All (A2A) 按attention head维度重分布QKV，inter-node使用Point-to-Point (P2P) 传输KV chunks；(3) 推理模式——所有GPU并发参与计算（vs HuggingFace Pipeline仅1 GPU活跃），动态管理逐步变化的token和position encoding，检测终止信号。MM-SP通过monkey-patching方式集成到HuggingFace Transformers，无需修改核心代码。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# MM-SP Training Mode with 2D-Attention (sp_size=8, mesh=4x2)
# sp_group: (A2A_size=4 intra-node, P2P_size=2 inter-node)

def mm_sp_training_step(batch, sp_rank, sp_mesh):
    # === Two-Stage Sharding ===
    # Stage 1: Distribute frames evenly for vision encoding
    n_frames = batch.frames.shape[0]  # e.g., 256
    local_frames = distribute_evenly(batch.frames, sp_mesh.world_size)
    # Each GPU: 256/8 = 32 frames → balanced vision workload
    
    vis_feats = vision_encoder(local_frames)  # [32 frames × 256 tokens = 8192]
    
    # Stage 2: Aggregate & re-shard by token count
    all_vis_feats = all_gather(vis_feats, sp_group)  # [65536 vis tokens]
    full_seq = concat([all_vis_feats, batch.text_tokens])  # + T text tokens
    local_tokens = balanced_shard_by_tokens(full_seq, sp_rank, sp_mesh.world_size)
    # Pad with dummy tokens for even division; mask in loss
    
    # === LLM Forward with 2D-Attention ===
    for layer in llm.transformer_layers:
        local_tokens = layer.ffn(local_tokens)  # local computation
        
        # 2D-Attention
        Q, K, V = layer.project_qkv(local_tokens)
        
        # A2A: intra-node (NVLink 900 GB/s)
        Q = all_to_all(Q, scatter_dim='head', gather_dim='seq', group=A2A_group)
        K = all_to_all(K, scatter_dim='head', gather_dim='seq', group=A2A_group)
        V = all_to_all(V, scatter_dim='head', gather_dim='seq', group=A2A_group)
        
        # P2P: inter-node KV transfer (InfiniBand 50 GB/s)
        for step in range(P2P_size - 1):
            send(K_chunk, dst=(sp_rank+1) % P2P_size, group=P2P_group)
            recv(K_remote, src=(sp_rank-1) % P2P_size, group=P2P_group)
            K = concat([K, K_remote])
            # Same for V
        
        # Local attention with full KV
        attn_out = FlashAttention2(Q, K, V, causal_mask=True)
        
        # Reverse A2A
        attn_out = all_to_all(attn_out, scatter_dim='seq', gather_dim='head', group=A2A_group)
        
        local_tokens = local_tokens + attn_out
    
    loss = CE(llm.lm_head(local_tokens), labels)
    return loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MM-SP通过Triton实现attention kernel，集成Flash-Attention2作为本地注意力后端。使用FSDP进行数据并行sharding。通信层使用NCCL的A2A和P2P primitives。推理模式下额外管理动态tensor（input tokens + position encodings的变化）和进程终止检测。支持扩展到256 GPU（32个8-GPU节点），支持2M+ token上下文长度。开源在github.com/NVlabs/VILA/tree/main/longvila。适用于需要处理超长视频序列（数百到数千帧）的VLM训练和推理场景。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## 2D-Attention (for Sequence Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
2D-Attention是MM-SP系统中的核心通信-计算调度机制，将序列并行度分解为两个维度：(1) Head Parallel维度（Ulysses-style All-to-All）——在intra-node内使用A2A按attention head维度重分布Q/K/V tensors，利用NVLink 900 GB/s高带宽；(2) Ring Parallel维度（ZigZag-style P2P）——跨节点使用P2P ring传输KV chunks，利用InfiniBand 50 GB/s互联。构建N_head × N_ring的二维通信mesh（如8度SP=4×2 mesh）。与纯Ring Attention相比，2D-Attention将高频的A2A通信限制在intra-node（快通道），低频的P2P通信仅用于跨节点（慢通道），避免Ring Attention中"所有链路都用P2P"导致的18×带宽差异利用不充分和通信占用SM资源问题。与DeepSpeed-Ulysses相比，2D-Attention的ring维度不受attention head数量限制，可扩展到任意GPU数量。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 2D-Attention Communication Schedule (sp_degree=8, mesh=4×2)
# 4 GPUs A2A group (intra-node), 2 GPUs P2P group (inter-node per rank)
#
# Ring Attention vs ZigZag RingAttn vs 2D-Attention comparison:
#
# RingAttention (sp=4, seq_len=8 tokens):
#   Step 1: Rank0[R0: tok0,1] Rank1[R1: tok2,3] Rank2[R2: tok4,5] Rank3[R3: tok6,7]
#            Compute attn(Q0,K0,V0)  Compute(Q1,K1,V1)  Compute(Q2,K2,V2)  Compute(Q3)
#            Send KV to R1           Send KV to R2       Send KV to R3       Send KV to R0
#   Step 2:  IDLE (no Q left!)       Compute(Q1,K[0,1])  Compute(Q2,K[1,2])  Compute(Q3,K[2,3])
#            # Rank0 idle after first round due to causal triangle
#
# ZigZag-RingAttn: reorder input sequence tokens to balance load
#   After reorder, all ranks have work across all steps
#   But still uses P2P for ALL communication (wastes intra-node NVLink)
#
# 2D-Attention (4 A2A × 2 P2P mesh):
# Step 1: A2A (intra-node, 4 GPUs, NVLink 900 GB/s)
#   Each GPU: Q [B, seq/8, num_heads, d] → all_to_all by head_dim → Q [B, seq/2, num_heads/4, d]
#   Communication: O(num_heads × d × seq/sp) ≈ 24MB/GPU
#   Time: ~27μs (NWLink bandwidth)
#
# Step 2: P2P Ring (inter-node, 2 GPUs, InfiniBand 50 GB/s)
#   GPU in node 0 sends KV to GPU in node 1, receives KV from node 1
#   Communication: O(num_heads/4 × d × seq/sp) ≈ 2MB/GPU  
#   Time: ~40μs (IB bandwidth)
#
# Step 3: Local FlashAttention2
#   Q_local against K_all,V_all (local + received from ring)
#   Attention mask: causal (triangular)
#
# Step 4: Reverse A2A to restore original head distribution
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于USP (Unified Sequence Parallelism, Fang & Zhao 2024) 和LoongTrain (Gu et al. 2024)的2D-SP思想，LongVILA首次将其扩展到多模态场景（处理复杂attention mask和变长输入序列）。A2A使用NCCL all-to-all，P2P使用NCCL send/recv。本地attention使用Flash-Attention2。Attention kernel使用Triton实现（论文指出port到C++可获进一步加速）。适用于超长序列（256K+ tokens）的多节点VLM训练，在256 GPU上支持2M+ tokens上下文，比Ulysses扩展约8×，比Megatron-LM CP/CP+TP显著更高效。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## Two-Stage Sharding for VLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
两阶段Sharding是MM-SP系统解决VLM中视觉模态和文本模态处理异构性的负载均衡策略。问题来源：在text-only LLM的序列并行中，所有token由同一tokenizer产生，可直接均分。但在VLM中，视觉数据首先由视觉编码器（ViT+投影器）处理并将placeholder token（如\\<img\\>）扩展为多个真实token（每帧约256 tokens）。如果简单地将placeholder tokens等同text tokens进行切分，会导致视觉编码阶段GPU负载不均。两阶段Sharding的解决方案：(1) Stage1（视觉编码阶段）——将所有帧均分到SP group内的各GPU，每GPU独立执行视觉编码，负载均衡；(2) Stage2（LLM解码阶段）——将所有视觉特征和文本token汇总后，按token数量在sequence维度均分（含dummy token padding确保均匀可分），实现LLM解码的负载均衡。此重分布仅在训练开始时执行一次，开销极小。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Two-Stage Sharding Pseudocode
def two_stage_sharding(frames, text_tokens, sp_rank, sp_size):
    """
    frames: [N_frames, H, W, C] where N_frames varies per sample
    text_tokens: [T]
    """
    # === Stage 1: Per-Image Balanced Distribution ===
    n_local_frames = N_frames // sp_size
    start = sp_rank * n_local_frames
    local_frames = frames[start : start + n_local_frames]
    # Each GPU gets exactly N_frames/sp_size frames
    # Encoding workload is balanced because each frame → ~256 tokens
    
    vis_features = vision_encoder(local_frames)  # [n_local_frames * 256, d]
    
    # === Stage 2: Global Aggregation + Per-Token Balanced Sharding ===
    # All-gather vision features
    all_vis_features = all_gather(vis_features, dim=0)  # [N_frames*256, d]
    
    # Concatenate with text tokens
    full_sequence = concat([all_vis_features, text_embedding(text_tokens)], dim=0)
    total_len = full_sequence.shape[0]  # e.g., 65536 + 2000 = 67536
    
    # Balanced sharding by token count
    tokens_per_rank = ceil(total_len / sp_size)
    # Pad with dummy tokens to make evenly divisible
    padded_len = tokens_per_rank * sp_size
    padded_sequence = pad(full_sequence, (0, padded_len - total_len))
    
    start = sp_rank * tokens_per_rank
    local_tokens = padded_sequence[start : start + tokens_per_rank]
    
    # Adjust labels to ignore padded tokens
    labels = adjust_labels_for_padding(labels, padded_len - total_len)
    
    return local_tokens, labels
```
Ablation结果（Table 5）：在8 GPUs上，long captioning任务中两阶段比一阶段快7%（1.12s vs 1.20s/iter），收益在更长captioning任务中更显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
两阶段Sharding在实现上需要注意：(1) Stage1的视觉编码器输出特征维度（每帧token数）需提前知道以做均匀分帧（VILA中每帧固定256 tokens）；(2) dummy token padding需在loss计算中mask掉（修改labels为ignore_index）；(3) 重分布通信仅一次（训练开始时），开销<1% total time。适用于所有包含视觉编码器+LLM解码器的VLM架构的序列并行训练场景。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## Sequence Parallelism (SP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sequence Parallelism (SP) 是一种分布式训练技术，将单个超长训练序列沿sequence维度切分到多个GPU上，每个GPU持有序列的一段（chunk），通过通信collective交换中间结果完成attention等跨token操作。与Data Parallelism（复制模型，每个GPU处理不同样本）和Tensor/Pipeline Parallelism（切分模型参数）不同，SP专门解决单序列超长导致单GPU显存不足的问题。常见SP方法包括：(1) Ring-style SP——使用P2P通信在GPU ring中传递KV blocks（RingAttention, LightSeq, ZigZag-RingAttn）；(2) Ulysses SP——使用All-to-All在head维度和sequence维度间切换分片（DeepSpeed-Ulysses）；(3) 2D/混合SP——结合Ring和Ulysses的优点（USP, LoongTrain, MM-SP）。LongVILA的MM-SP是首个为VLM设计的SP系统。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Sequence Parallelism vs Data Parallelism (8 GPUs, seq_len=256K)  
#
# Data Parallelism (FSDP):
#   Each GPU: full sequence [256K tokens] → OOM!
#   FSDP only shards model params/grads/optimizer states
#   Each GPU must hold full activations for its full sequence
#
# Sequence Parallelism (sp_degree=8):
#   Each GPU: local chunk [32K tokens]
#   Attention: need KV from other GPUs for full context
#
# Ring-SP Communication Pattern (sp=4):
#   Initial: Q0[0:32K], Q1[32K:64K], Q2[64K:96K], Q3[96K:128K]
#   Step 1: GPU0→1 K0,V0; GPU1→2 K1,V1; GPU2→3 K2,V2; GPU3→0 K3,V3
#           Compute attn with local Q + received K,V
#   Step 2: GPU0→1 K3,V3; GPU1→2 K0,V0; GPU2→3 K1,V1; GPU3→0 K2,V2
#           Compute attn with updated K,V
#   Step 3: GPU0→1 K2,V2; GPU1→2 K3,V3; GPU2→3 K0,V0; GPU3→0 K1,V1
#           Compute attn with updated K,V
#   Total P2P: 3 rounds per attention layer
#   Bandwidth: all P2P uses same interconnect (no hierarchy awareness)

# MM-SP 2D-Attention (sp=8, 4×2 mesh):
#   A2A group (intra-node): 4-way A2A reshuffles QKV by head dim → O(NVLink)
#   P2P group (inter-node): 2-way P2P ring transfers KV only → O(InfiniBand)
#   A2A happens once per layer; P2P happens once (sp_ring=2 means 1 transfer)
#   Total: 1 A2A + 1 P2P round per attention layer (for ring_size=2)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SP的实现依赖于底层通信库（NCCL）提供的collective primitives（P2P send/recv, All-to-All, All-Gather）。对于attention计算，需要在QKV projection之后、SDPA之前插入通信操作来汇聚完整K/V。对于VLM，还需处理视觉token的特殊性（placeholder扩展、变长序列等）。MM-SP通过Triton实现kernel，monkey-patch到HuggingFace Transformers。适用于：(1) 超长序列训练（video VLM 256K+ tokens）；(2) 单batch size=1但序列极长的场景；(3) 需要超越attention head数量限制的扩展（vs Ulysses）。

涉及论文标题：
- LongVILA__Scaling_Long-Context_Visual_Language_Models_for_Long_Videos

## FlashAttention（融合注意力Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashAttention 是由 Tri Dao 等人在 NeurIPS 2022 提出的 I/O-aware 精确注意力算法，通过 kernel fusion 和 tiling 将 attention 计算保留在 GPU SRAM 中，避免构建完整 attention map 并写入 HBM，从而显著减少 HBM I/O 并加速 attention 计算。后续版本 FlashAttention-2 (2023) 进一步优化了 warp 调度和 parallelism；FlashAttention-3 (2024) 利用 Hopper 架构的 TMA + WGMMA + warp specialization。

核心机制：(1) Online softmax tiling——将 QKV 分 tile 加载到 SRAM，在 tile 间传递 running max/sum 实现增量 softmax 归一化，无需物化完整 S = QK^T 矩阵；(2) Kernel fusion——QKV projection、attention、dropout、residual add 融合为单 kernel；(3) Backward recomputation——不存储 attention map，通过 softmax normalization statistics 重计算。

对 token compression 的约束：FlashAttention **不暴露中间 attention map**——Sij/Pij 从未离开 SRAM。依赖 attention map 的 token 重要性评分方法（EViT、BAT、vid-TLDR、FastV、SparseVLM、PDrop）因此与 FlashAttention 不兼容，必须 fallback 到标准 attention 实现，导致 GPU 峰值显存超过未压缩模型（FastV 增 3.7%，SparseVLM 在视频场景增 54.8%）。绕过策略：(1) Representation Shift 使用 MLP 前后的 L2 距离；(2) V2Drop 使用相邻 LLM 层之间的 token-wise L2 variation（||f_i^(l) - f_i^(l-1)||_2），仅需 3MD' FLOPs（三层总开销约 21M FLOPs，完整前向的 0.002%），天然兼容 FlashAttention，GPU 峰值显存与 random dropping 相同。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FlashAttention Forward (single head, tiled online softmax)
for i in range(Tr):          # Q tiles
    Qi = load_tile(Q, i)      # [Br, d] HBM→SRAM
    mi = -inf; li = 0; Oi = zeros(Br, d)
    for j in range(Tc):       # K, V tiles
        Kj, Vj = load_tile(K, j), load_tile(V, j)
        Sij = Qi @ Kj^T       # [Br, Bc], on SRAM
        m_new = max(mi, rowmax(Sij))
        Oi = Oi * exp(mi - m_new); li = li * exp(mi - m_new)
        Pij = exp(Sij - m_new)
        li = li + rowsum(Pij); Oi = Oi + Pij @ Vj
        mi = m_new
        # Pij 不写回 HBM——token importance 方法无法获取
    Oi = Oi / li[:, None]; store(Oi); store(mi, li)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：`pip install flash-attn` → `flash_attn_func(q, k, v)`；或 PyTorch 2.0+ `scaled_dot_product_attention` 自动调用。vLLM/TGI 等框架通过 flag 启用。对 ViT/Video Transformer 同样有效——UMT-B 上 2.7× speedup（NVIDIA RTX A6000）。与不需要 attention map 的 token pruning（如 Representation Shift）叠加使用可实现乘法级加速（~5.5× total）。

涉及论文标题：
- Representation_Shift__Unifying_Token_Compression_with_FlashAttention
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding
- XStreamVGGT__Extremely_Memory-Efficient_Streaming_Vision_Geometry_Grounded_Transformer_with_KV_Cache_Compression

VideoNSA 将 FlashAttention-2 作为 baseline 的 attention kernel（用于 Qwen2.5-VL-7B 的 dense GQA 推理）。NSA 的三个稀疏支路的 attention 操作在实现上复用 FlashAttention 的 batch GEMM + online softmax kernel，但仅对稀疏选定的 KV subsets 计算（而非完整 KV cache）。VideoNSA 的 delay 分析（Figure 6）显示：compression branch 在长 context（128K）下是主要瓶颈（因需处理所有压缩 blocks），而 selection/swa branch 延迟仅小量增长。FlashAttention 的 kernel fusion 与 NSA 的 block-level sparse attention 的 tile 大小需对齐以获得最佳硬件效率——NSA 的 block size=64 与 FlashAttention 的 tile 规格兼容。论文指出进一步优化 compression branch 的 kernel design 和 memory efficiency 是未来方向。
