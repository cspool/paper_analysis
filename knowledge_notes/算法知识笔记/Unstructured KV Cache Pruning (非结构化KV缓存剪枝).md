## Unstructured KV Cache Pruning (非结构化KV缓存剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Unstructured KV Cache Pruning 是一种对 LLM 推理中 Key-Value 缓存进行压缩的技术：移除 KV cache 矩阵中任意位置的单个元素（标量），而非按 channel、token 或 block 等结构化单元剪枝。与结构化剪枝（如 ThinK 的 per-channel Key cache 剪枝，每次移除整个 channel）不同，非结构化剪枝不对稀疏 pattern 施加任何几何约束——每个元素独立判断是否保留。

Mustafar 论文核心发现：Key cache 非结构化剪枝在 70% 稀疏度下精度（LongBench avg 41.55）远优于 ThinK 50% 结构化剪枝（38.53）。Value cache 非结构化剪枝突破结构化剪枝 30% 稀疏度上限，在 70% 稀疏度下保持精度（42.78 vs Dense 43.19）。

非结构化优于结构化的原因：(1) Key cache 虽有 channel-wise outliers，但 outlier channel 内部并非所有元素都有用——结构化整 channel 丢弃损失有价值元素，整 channel 保留携带冗余；(2) Value cache 元素分布均匀无 channel outliers，结构化剪枝无法识别有效 channel 导致 30% 实用上限；(3) 元素级剪枝实现比 channel 级更精确的取舍。

与 2:4 Semi-structured Sparsity 对比：2:4 是 NVIDIA Sparse Tensor Core 的模式——每连续 4 元素恰好保留 2 个，全局 50% 稀疏度。Mustafar 实验显示同等 50% 稀疏度下非结构化精度优于 2:4（LongBench avg 42.65 vs 40.89），因为 2:4 对局部 pattern 仍施加约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 非结构化KV Cache剪枝pipeline
# 输入: K_cache, V_cache ∈ R^{Tx d}, 稀疏度 s, 局部窗口 W=32

# Step 1: Per-token magnitude-based pruning
for i = 1..T-W:
    abs_K = |K_cache[i, :]|          # shape [d]
    thresh_K = np.partition(abs_K, d*s)[d*s]
    mask_K[i] = (abs_K >= thresh_K)

# Step 2: Value cache同理
for i = 1..T-W:
    abs_V = |V_cache[i, :]|
    thresh_V = np.partition(abs_V, d*s)[d*s]
    mask_V[i] = (abs_V >= thresh_V)

# Step 3: 最近W个token保持dense (mask = all 1)

# Step 4: Apply及bitmap压缩
K_sparse = K_cache * mask_K
K_compressed = bitmap_compress(K_sparse, mask_K)  # tile=1x64
```

术语一般如何实现？如何使用？

实现需求：(1) 逐元素排序/top-k选择，计算开销 O(Td)，在 prefill 结束后批量执行；(2) 稀疏存储需 bitmap 或 CSR/COO 格式——irregular pattern 无法通过简单减少 matrix dimension 压缩；(3) 需配合特殊 kernel（Mustafar SpMV、FlashLLM SpMM）将内存节省转为计算加速；(4) 50% 稀疏度实际压缩比 65%（15% metadata overhead），70% 为 45%。适用场景：长上下文 LLM decode（memory-bound），prefill 仍用 FlashAttention。开源：https://github.com/dhjoo98/mustafar。

涉及论文标题：
- Mustafar__Promoting_Unstructured_Sparsity_for_KV_Cache_Pruning_in_LLM_Inference
