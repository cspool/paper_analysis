## Group Block Sparse Attention (GBSA) with Shared Grouping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GBSA 是 ReSA 在 Quest block-sparse attention 基础上为 GQA 模型优化的变体。利用 GQA 结构——同一 GQA group 内多个 query heads 共享同一组 KV heads——在 group 级别统一做 block selection，组内所有 query heads 复用相同 selected block indices。关键步骤：(1) 对 GQA group 内所有 query heads 做平均池化得 q_pool；(2) 仅用 q_pool 与 block descriptors 计算一次 similarity scores；(3) 选 top-n block indices；(4) 组内所有 heads 共享这组 indices 做 sparse attention。

从算法pipeline角度拆解术语：

```
for each GQA group j in 0..h_kv:
    q_pool = mean(Q[j, :, :, :], dim=0)  // g query heads pooling
    // 仅一次 block selection per group
    selected = top_n([Σ_d max(q_pool[d]×k_max_i[d], q_pool[d]×k_min_i[d]) for i in 0..M])
    // g query heads 共享 selected blocks
    for each head q in group j:
        o = softmax(q @ K[j,selected,:]^T / √d) @ V[j,selected,:]
```

与 per-head block selection 相比，GBSA 的 block selection 计算从 O(h_query × M × d) 降至 O(h_kv × M × d)。Qwen2.5 7B 配置下（28 query heads / 4 KV heads），head 维度减少 7×。同一 SM 上为多个 query head 加载的 KV block 数据可在 warp/thread 间通过 shared memory 复用。

术语一般如何实现？如何使用？

每个 GQA group 固定分配到一个 SM，SM 内为 g 个 query heads 计算 sparse attention。KV block 数据加载一次后在 SM 内多 warp 间共享。Block indices 在同 group 多 head 间直接复用。ReSA 使用 TileLang 实现该 kernel。

涉及论文标题：
- Rectified Sparse Attention

---
