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
