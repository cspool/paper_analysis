## Layer-wise KV Cache Sharing (层间KV缓存共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-wise KV Cache Sharing 是一种跨 Transformer 层的 KV cache 压缩方法，通过在某些层的推理过程中跳过本层 KV cache 的计算，直接复用（拷贝）其他已计算层的 KV cache，从而减少 KV cache 的计算量和存储占用。与 intra-layer 压缩（在单层内通过 token pruning、量化或 channel shrinking 减少 KV cache）正交，layer-wise 共享从"层间冗余"的角度出发，利用了不同层 KV cache 之间可能存在的可替代性。

KVSharer 论文首次提出了一种无需额外训练的即插即用 layer-wise KV cache 共享方法。该方法基于一个反直觉的发现：共享不相似（dissimilar）的 KV cache 比共享相似的 KV cache 能更好地保持模型性能。这与传统参数共享/注意力共享中"相似度越高共享效果越好"的直觉相反。

从算法pipeline角度拆解术语：

**KVSharer 的 Layer-wise KV Cache Sharing 流程**：

```
// 阶段一：离线策略搜索（Algorithm 1，约60秒/模型）
输入: 预训练LLM M, 目标共享层数 C, 校准数据集 D, 相似度阈值 T
输出: 共享策略 Z

1. 在 D 上运行 M，保存每层 KV cache
2. 将每层 keys 和 values 分别 flatten → 1D向量，取平均作为该层 KV cache 表示
3. 计算任意两层之间的欧氏距离并按降序排列
4. Z ← ∅, P ← 0
5. for each (src, dst) in 降序排列:
       Z ← Z ∪ {(dst层 KV cache ← src层 KV cache)}
       // 输出端被输入端替换（输入端更敏感，不可反向）
       M_tmp ← 应用 Z 的模型
       s ← AvgCosineSim(M_tmp.last_hidden_state, M.last_hidden_state, D)
       if s ≤ T (0.5):  Z ← Z \ {(dst, src)}
       else:            P ← P + 1
       if P == C: return Z

// 阶段二：在线推理
for l in 1..num_layers:
    if l 是被替换层:
        K_cache[l] = K_cache[src_layer]  // 直接拷贝
        V_cache[l] = V_cache[src_layer]
    else:
        K_cache[l], V_cache[l] = compute_KV(l, x)
    output = attention(Q, K_cache[l], V_cache[l])
    output = FFN(output)
```

术语一般如何实现？如何使用？

KVSharer（https://github.com/yangyifei729/KVSharer）以即插即用方式实现，在 HuggingFace Transformers 的 forward pass 中根据预搜索策略 Z 修改每层的 KV cache 计算逻辑。与 FlashAttention 和 GQA/MHA 兼容。压缩率通过 C 控制：12.5%~37.5%。25% 压缩率下保持 >90% 性能。策略搜索仅需约 60 秒（4×A100），一次搜索可通用于所有下游任务。

涉及论文标题：
- KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 的 SVD-based Layer-wise KV Cache Sharing**：与 KVSharer 直接拷贝整层 KV-Cache（全等共享）不同，xKV 通过**跨层 SVD** 实现更精细的层间共享——不直接复用某一层的完整 KV-Cache，而是从多层的拼接 KV-Cache 中提取**共享的低秩基 A**，每层保留独立的低秩重构矩阵 B_ℓ_i。这种方式比全等共享更灵活（每层保留其特定信息），比单层独立压缩更高效（共享跨层公共子空间）。xKV 在 8× 压缩比下通过 xKV-4（4 层一组）实现 87.8% avg accuracy（vs KVSharer 类方法通常在 1.2x 压缩就出现明显退化）。

---
