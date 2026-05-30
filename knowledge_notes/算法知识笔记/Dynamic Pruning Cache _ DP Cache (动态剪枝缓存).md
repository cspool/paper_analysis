## Dynamic Pruning Cache / DP Cache (动态剪枝缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Pruning Cache (DP Cache) 是 DyCoke 提出的一个辅助 KV cache 结构，用于在 decoding 阶段存储被动态剪枝机制暂时移除的 visual token 的 K/V 对。与传统 one-shot pruning 的"永久丢弃"策略不同，DP Cache 使得 token 可以在后续 decoding 步骤中被召回：被剪枝的 token 进入 DP Cache（而非直接释放），当模型注意力重新关注到这些 token 时（通过每隔 N 步重新计算 cross-attention 检测），它们会被动态加回主 KV cache。同时，KV cache 中注意力下降的 token 也会被移回 DP Cache。KV Cache 和 DP Cache 之间形成双向流动：KV→DP（剪枝）和 DP→KV（召回）。这种设计解决了 one-shot pruning 无法适应 decoding 过程中注意力分布变化的根本缺陷。DP Cache 中 token 不参与当前步骤的 attention 计算，因此不消耗 attention 的 O(n²) 计算成本。DyCoke 消融实验证明去除 DP Cache 后 VideoDC benchmark 性能显著下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DP Cache 在 DyCoke 两阶段压缩中的位置和运转流程：
```
# === DP Cache 运转流程 ===
# 初始状态: KV_cache = {TTM压缩后的visual KV + text KV}
#         DP_cache = {}  (空)

# 首次动态剪枝 (t=1)
A = Softmax(Q_pred K_visual^T / sqrt(D))
keep_idx = top_P_percent(A, P)   # 保留 top-P%
prune_idx = complement(keep_idx)  # 剩余低注意力 token

# 双向分流:
KV_cache[visual]  = KV_full[visual][keep_idx]   # 高注意力 → KV cache
DP_cache           = KV_full[visual][prune_idx]  # 低注意力 → DP cache (保存!)

# 后续解码 (每 N=1 步或注意力变化时)
if cosine_sim(attn_prev, attn_curr) < sim_threshold:
    # 重新评估：将 DP cache 中的 token 与 KV cache 中的 token 联合评估
    A_full = Softmax(Q_pred [K_kv; K_dp]^T / sqrt(D))
    
    # KV cache 更新
    new_keep = top_P_percent(A_full, P)
    new_prune = complement(new_keep)
    
    KV_cache[visual] = [K_kv; K_dp][new_keep]   # 包含从 DP 召回的高注意力 token
    DP_cache         = [K_kv; K_dp][new_prune]   # 包含从 KV 移出的低注意力 token

# 每步仅用 KV cache 中的 token 计算 attention
output = attention(Q_pred, KV_cache)
```
关键设计：KV cache ↔ DP cache 的双向流动性。DP cache 中 token 不参与 attention 计算以节省 FLOPS，但保留完整的 K/V 以便召回。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DP Cache 直接使用与主 KV cache 相同的 key/value 张量格式存储，实现为 PyTorch tensor 的 gather/scatter 操作（基于索引选择）。由于仅在每隔 N 步重新计算 cross-attention（当相邻迭代注意力分布余弦相似度低于阈值时），DP cache 的索引管理开销可控。DyCoke 使用 L=3 层（而非所有层）进行动态评估，进一步降低开销。与 Flash Attention 兼容。类似机制：SparK 的 channel-level recovery（通过存储分布模式重建被剪枝通道）、LeanKV 的混合精度 downgrade 路径。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
