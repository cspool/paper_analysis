## Intra-Group Layer KV Cache Indices Sharing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Intra-Group Layer KV Cache Indices Sharing 是 WindowKV 的计算效率策略：将 m 层分为 H=m/γ 组，仅每组首层 l_g 执行 window selection 获得 indices I_{lg}，组内其余层直接复用，省去重复的 attention 计算。理论基础是 LLM 相邻层 attention 分布相似（Ma et al., 2024; Liu et al., 2025, ChunkKV），并通过同组内层间 KV cache index 的 Jaccard similarity 实验验证。

消除实验：γ=1（无共享，32.13）vs γ=7（共享，32.75 on Qwen2.5）——适度共享因预算更均匀分布而略有提升；γ=14 时 budget 过于均匀破坏金字塔结构（27.83）。LLaMA3-8B 最优 γ=8。

公式：$H = m/\gamma$, $\mathbb{I}_{l_{h\gamma}}$ 仅首层计算，$\mathbb{I}_{l_{h\gamma+k}} = \mathbb{I}_{l_{h\gamma}}$ for $k \in [1, \gamma-1]$。计算开销 O(m·n²) → O(H·n²) = 1/γ ×。

术语一般如何实现？如何使用？

实现：在 Transformer forward pass 中 `if layer_idx % γ == 0` 则执行完整 window selection，否则复用上一首层 indices 对 KV cache 做 gather 操作。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
