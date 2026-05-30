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
