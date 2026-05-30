## Pre-RoPE / Post-RoPE Space for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pre-RoPE 和 Post-RoPE 空间是 RoPE 位置编码前后的向量空间区分。Pre-RoPE 空间：Q/K 向量在 RoPE 旋转之前的空间，仅编码内容信息，不受位置影响。Post-RoPE 空间：Q/K 向量经 RoPE 旋转（乘以 e^{iωp}）后的空间，编码"内容+位置"混合信息。

这一区分对 KV cache 压缩方法至关重要。TriAttention 首次明确指出 post-RoPE 方法的系统性限制：query 经 RoPE 旋转后方向随位置连续变化，只有最近的 query 具有"当前"朝向，导致观察窗口极小（约 25 个 query 最优）——这是 post-RoPE 方法固有的，无法通过增加窗口大小解决（Zhang et al., 2025 确认：增加到 25 个 query 后性能下降）。

Pre-RoPE 空间不受位置旋转影响——Q/K 围绕固定中心聚集（Q/K Concentration 现象），跨位置稳定。TriAttention 回到 pre-RoPE 空间，利用 Q 中心替代未来 query 预测 attention 模式，完全绕过观察窗口限制。

术语一般如何实现？如何使用？

Pre-RoPE 向量的获取：在模型 attention layer 中 RoPE 旋转之前截取 Q/K 中间表示。在 vLLM 中通过 monkeypatch attention forward 实现。使用场景：任何需要考虑 Q/K 方向信息且不希望受位置编码污染的注意力分析——KV 压缩（TriAttention）、attention head 功能分类、模型诊断。Pre-RoPE 空间的 Q/K 向量跨位置稳定，是分析 head 语义功能比 post-RoPE 更优的信息源。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---
