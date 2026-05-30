## concat_reorder Operator (dKV-Cache 的 KV 重排操作符)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

concat_reorder 是 dKV-Cache 论文中为扩散语言模型 (DLM) 的 KV 缓存设计的底层操作符，用于高效实现非连续位置的 Key/Value gather 和 scatter 操作。在标准 KV Cache (AR) 中，缓存 token 是连续的（位置 1→t），只需简单的 concat 追加操作。但 DLM 中解码位置是任意的，缓存 token 分散在整个序列的非连续位置，导致每步需要多次 gather（收集缓存 token 的 K/V）和 scatter（将新 K/V 散播回原位置）——这些非连续内存访问在 GPU 上效率极低。

concat_reorder 的核心思想是将索引操作从 K/V 矩阵层级（形状 [B, L, D]，GPU 上昂贵的大矩阵 gather/scatter）**下移**到 token 层级（形状 [B, L]，更轻量的序列重排）。具体做法：每步将序列重新排列，使所有缓存 token 位于左侧连续排列、未缓存 token 位于右侧，从而用简单的 Concat 和 Slice 操作替代 gather/scatter。token 位置编码随重排同步调整（每步一次、跨层共享），额外开销可忽略。

从kernel调度角度拆解术语。

**concat_reorder 的运作流程**（伪代码）：

```
// 步 t-1 结束后：
// 缓存 token 位置集合：I \ M_{t-1} = [2, 4, 5, 7]（分散的）
// 掩码 token 位置集合：M_{t-1} = [0, 1, 3, 6]
// 总序列长度 L = 8

// 步 t 开始：
1: Gather:
   K_cache = K_{t-1}[I \ M_{t-1}]    // 从缓存 K 中索引取出 [B,4,D]
   V_cache = V_{t-1}[I \ M_{t-1}]    // 从缓存 V 中索引取出 [B,4,D]
   // 仅一次 gather，O(BLD)

2: Reorder sequence:
   x' = [x[2], x[4], x[5], x[7], x[0], x[1], x[3], x[6]]
   // 缓存 token 在左、掩码 token 在右

3: Reorder PE:
   PE' = [PE[2], PE[4], PE[5], PE[7], PE[0], PE[1], PE[3], PE[6]]
   // 仅一次，跨所有层共享

4: Transformer forward (仅掩码 token):
   K_new, V_new = Transformer(x'[4:])  // 仅计算 M_{t-1} 部分的 4 个 token
   
5: Concat (而非 gather/scatter):
   K_all = Concat([K_cache, K_new], dim=1)  // [B,8,D]，简单拼接！
   V_all = Concat([V_cache, V_new], dim=1)

6: Attention:
   Q_new = Q(x'[4:])  // 仅掩码 token 的 Query
   output = Attention(Q_new, K_all, V_all)  // 双向注意力

7: Update mapping and cache:
   // 步 t 结束后新的缓存集 M_t = [0, 1, 3]（又有 3 个 token 被解码）
   // I \ M_t = [2, 4, 5, 7, 0, 1, 3]（7 个缓存 token）
   // 新的重排映射: [0, 1, 2, 3, 6, 4, 5] (从当前 [2,4,5,7,0,1,3,6] 中索引)
   K_cache_next = Reorder(K_all, mapping)   // 提取 I\M_t 对应的行
   V_cache_next = Reorder(V_all, mapping)
```

**token 级 vs 矩阵级操作的开销对比**：

| 操作 | 原始方法 (矩阵级) | concat_reorder (token 级) |
|------|---------------------|---------------------------|
| Gather K cache | gather([B,L,D], index) → O(BLD) 随机读取 | gather([B,L], index) → O(BL) token 重排 |
| Scatter new KV | scatter([B,|M|,D], index) → O(B|M|D) 随机写入 | concat([B,|cache|,D], [B,|M|,D]) → 连续拼接 |
| 内存访问 | 大量非连续、小粒度的 HBM 事务 | 连续块操作，适合 DRAM burst |
| PE reorder | 论文未明确说明 | O(BL) 简单索引，一次跨层共享 |

术语一般如何实现？如何使用？

在 dKV-Cache 开源实现（https://github.com/horseee/dKV-Cache）中，concat_reorder 作为修改版模型 forward 函数的核心组件：

```python
# 概念实现（简化版）
def concat_reorder_step(x, pe, K_cache, V_cache, M_prev, M_curr):
    """
    x: [B, L] token IDs
    pe: [B, L, D] positional encodings
    M_prev: set of masked positions at step t-1
    M_curr: set of masked positions at step t (unknown until after forward)
    """
    # Step 1: Reorder sequence (cache left, masked right)
    cache_idx = sorted(set(range(L)) - M_prev)
    masked_idx = sorted(M_prev)
    reorder_idx = cache_idx + masked_idx
    x_reordered = x[:, reorder_idx]
    pe_reordered = pe[:, reorder_idx]
    
    # Step 2: Compute only masked tokens
    x_masked = x_reordered[:, len(cache_idx):]
    K_new, V_new = transformer_layers(x_masked, pe_reordered[:, len(cache_idx):])
    
    # Step 3: Concatenate cached with new (NO gather/scatter)
    K_full = torch.cat([K_cache, K_new], dim=1)  # simple concat!
    V_full = torch.cat([V_cache, V_new], dim=1)
    
    # Step 4: Attention with full K/V
    Q_new = compute_query(x_masked)
    output = scaled_dot_product_attention(Q_new, K_full, V_full)
    
    # Step 5: Update cache for next step
    # Determine new mapping from current order to next cache set
    new_cache_idx = sorted(set(range(L)) - M_curr)
    mapping = [reorder_idx.index(i) for i in new_cache_idx]
    K_cache_next = K_full[:, mapping, :]  # reorder to get next cache
    V_cache_next = V_full[:, mapping, :]
    
    # Step 6: Scatter output back to original positions
    output_full = scatter(output, M_prev, L)  # token-level scatter
    
    return output_full, K_cache_next, V_cache_next
```

实际部署考量：(1) 该方法引入额外但可接受的 overhead——位置编码重排跨层共享，仅每步一次；(2) batch size=1 时 gather/reorder 开销可能使缓存加速被 offset，但 batch size≥2 时加速显著；(3) 对于 Dream 模型，Un&Right-Shift 缓存策略与 concat_reorder 不兼容（因位置偏移逻辑冲突），需退化为 Un-Shift 策略；(4) concat_reorder 仍有优化空间——论文承认"仍引入一些 overhead"，未来可结合 CUDA kernel fusion 或 Triton 进一步优化索引操作。

涉及论文标题：
- dKV-Cache: The Cache for Diffusion Language Models
