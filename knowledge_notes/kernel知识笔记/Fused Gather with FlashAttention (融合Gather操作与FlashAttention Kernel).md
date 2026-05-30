## Fused Gather with FlashAttention (融合Gather操作与FlashAttention Kernel)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Fused Gather with FlashAttention是HATA中将sparse attention的gather操作（根据indices从K/V cache中选择top-k entries）融合到FlashAttention kernel内部的技术。在非融合实现中，需要先将K_sparse=Gather(K_cache, Idx)和V_sparse=Gather(V_cache, Idx)的结果写入HBM，再由FlashAttention从HBM读回——导致冗余的HBM↔SRAM数据传输。通过kernel融合，FlashAttention在tiling过程中直接根据indices选择性加载所需的K/V tiles，消除了冗余的数据搬运。

在HATA的消融实验中，该优化单独贡献了23.8%的延迟减少。

从kernel调度角度拆解术语：

```
# Before (non-fused, 3 operations):
K_sparse = K_cache[Idx]              # Op 1: Gather K: HBM read → HBM write
V_sparse = V_cache[Idx]              # Op 2: Gather V: HBM read → HBM write
O = FlashAttention(Q, K_sparse, V_sparse)  # Op 3: HBM read → compute → HBM write
# Total HBM traffic: K_cache[s,d] + V_cache[s,d] (read) 
#                    + K_sparse[N,d] + V_sparse[N,d] (write then read)
#                    = 2*s*d + 4*N*d  bytes

# After (fused, 1 kernel):
O = FusedGatherFlashAttn(Q, K_cache, V_cache, Idx)
# Inside fused kernel (FlashAttention tiling):
# for each attention tile:
#     K_tile = GatherTile(K_cache, Idx[tile_start:tile_end])  # direct SRAM load
#     V_tile = GatherTile(V_cache, Idx[tile_start:tile_end])  # direct SRAM load
#     S_tile = Q_tile @ K_tile^T / sqrt(d)
#     P_tile = online_softmax(S_tile)
#     O += P_tile @ V_tile
# Total HBM traffic: K_cache[s,d] + V_cache[s,d] (selective reads only)
#                    = 2*N*d  bytes (only needed K/V tokens)
# Savings: eliminates K_sparse/V_sparse intermediate write + avoids loading 
#          irrelevant K/V tokens
```

术语一般如何实现？如何使用？

HATA实现（https://github.com/gpzlx1/HATA）将gather逻辑嵌入到FlashInfer的FlashAttention kernel中。基于FlashAttention-2的tiling框架，在每次tile迭代中根据indices计算实际需要加载的K/V全局内存地址，使用coalesced memory access模式按地址加载。对于GQA模型（多个query head共享KV），indices在共享KV head间共享，仅需计算一次gather地址。与FlashInfer框架完全兼容，用户可通过替换attention backend使用。

涉及论文标题：
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference
