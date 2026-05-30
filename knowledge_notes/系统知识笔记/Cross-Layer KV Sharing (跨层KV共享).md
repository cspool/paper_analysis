## Cross-Layer KV Sharing (跨层KV共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-Layer KV Sharing 是一种 KV cache 压缩技术。核心观察：相邻 Transformer 层的 KV cache 具有高度相似性（Brandon et al., 2024），因此让多个连续层共享同一组 K/V cache 可显著减少内存占用。Hymba 的具体实现：每 2 个连续层共享同一 KV cache（第 2i 和 2i+1 层共用），减少的参数量被重分配到其他模型组件，综合效果：缓存总量下降且模型质量提升（commonsense accuracy +0.60%）。配合 SWA 和 GQA，Hymba 将 8K cache 从 414.7MB 降至 39.4MB（10.5× reduction）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Hymba 中的 cross-layer KV sharing 缓存布局：

```
# 标准 Transformer: 每层独立 KV cache
Total cache = L x N x 2 x d_head x h_kv x 2bytes (FP16)

# Hymba + Cross-layer KV Sharing + SWA + GQA
# - 每 2 层共享（L/2 组）
# - 仅 3 层 global attention 存完整序列
# - 其余 29 层 SWA 仅缓存窗口 C=1024
# - GQA 减少 KV heads (h_kv < h_q)

# Hymba-1.5B 实例：
# 32 layers, 25 attn heads, 5 GQA groups
# 共享后约 16 组 KV cache
# 3 组 global: 3 x N x 2 x 64 x 5 x 2bytes
# 13 组 SWA: 13 x 1024 x 2 x 64 x 5 x 2bytes
# Total at N=8K: ≈ 79MB (vs Llama3-3B 918MB)
```

交互流程：推理时，layer 2i 写入 KV cache 共享 buffer → layer 2i+1 直接读取同一 buffer 执行 attention → 无需额外的 cache 副本或传输。Kernel 层面通过 shared memory pointer 实现零拷贝访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 共享粒度——Hymba 采用 2 层一组；Brandon et al. (2024) 研究了更多变体 (2/3/4 层共享)；(2) 适用条件——recall 和 commonsense accuracy 影响小甚至正面（节省参数可重分配到其他组件）；(3) 与其他技术互补——Hymba 组合 SWA + cross-layer KV sharing + GQA 达到最佳 cache 效率；(4) 架构约束——共享层需要相同 head 数和 head dim，需在设计初期规划。局限：仅在 attention heads 存在时生效（纯 SSM 模型无 KV cache）；共享后特定层的 attention 灵活性可能降低。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models
