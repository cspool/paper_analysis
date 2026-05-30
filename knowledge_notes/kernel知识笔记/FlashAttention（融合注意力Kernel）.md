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
