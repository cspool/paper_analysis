## TriAttention Scoring Function (TriAttention KV Scoring)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TriAttention 评分函数是 TriAttention KV cache 压缩方法的核心，结合两个互补信号（Eq 10）：$S(k, \Delta) = S_{\text{trig}}(k, \Delta) + S_{\text{norm}}(k)$。S_trig 捕获距离偏好（通过三角函数级数预测 key 在距离 Δ 处的 attention），S_norm 捕获低范数 key（S_norm = Σ_f (1-R_f)·E[‖q_f‖]·‖k_f‖，以 MRL 自适应加权）。

GQA 聚合策略：每个 KV head 被 G 个 query head 共享，产生 G 个不同尺度的评分。处理方式——per-head z-score normalize 后 max 聚合（Eq 12-13）——只要任一 query head 认为 key 重要就保留。Window-based Pruning：每 128 tokens 触发一次评分+pruning，保留 top-B keys。

消融验证（Qwen3-8B, KV budget 2048, AIME）：
- 去掉 S_trig（仅 Snorm）：AIME24 42.1% → 18.8% (-23.3%)
- 去掉 Snorm（仅 S_trig）：AIME24 45.8% → 40.4% (-5.4%)
- 去掉 R 加权：AIME25 32.9% → 28.7% (-4.2%)

术语一般如何实现？如何使用？

实现：集成到 vLLM 作为 plugin（通过 monkeypatch scheduler 和 worker），也支持 SGLang 和 MLX。校准离线处理一次。评分无需计算实际 attention（仅需 key 的 pre-RoPE 表示和离线 Q 中心），远低于 post-RoPE 方法（需计算完整 QK^T attention matrix）。在 AIME25 上匹配 Full Attention 准确率（40.8%）同时实现 2.5x throughput 或 10.7x KV memory reduction。代码开源：https://github.com/WeianMao/triattention。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---
