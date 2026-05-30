## One-Shot Token Pruning (一次性Token剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-Shot Token Pruning 是一种在 LLM/VLLM 推理的 prefilling 阶段，基于单次 token 重要性评估来永久性剪枝 visual token 的技术。代表方法：(1) FastV——在 LLM 的特定层（如第 5 层）计算 prompt token 对 visual token 的 attention score，保留 top-k% token 到 KV cache，此后 decoding 阶段不再改变；(2) LLaVA-PruMerge——基于 CLIP 视觉编码器的 attention score（而非 LLM 内部 attention）选择关键 visual token，一次性剪枝。两种方法的核心缺陷（DyCoke 的核心动机）：视频输入中，不同 decoding 步骤关注的视觉 token 不同（temporal attention shift），prefilling 阶段的 attention 分布与后续 decoding 需求不一致，一次剪枝后无法纠正错误判断。DyCoke 论文 Figure 2 通过可视化证明了这一现象：某些 frame 的 attention 在 decoding 后期显著上升，而 one-shot 方法可能已在早期将其剪除。One-shot pruning 的另一局限是：被剪枝 token 的 KV cache 永久丢失，无法被召回。DyCoke 通过引入动态剪枝 + DP Cache 解决问题：每一步可重新评估并调整剪枝集，被剪枝 token 保留在 DP Cache 中可召回。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FastV（典型 one-shot pruning）的流程：
```
# === One-Shot Token Pruning (FastV) ===
# 仅一次评估，永久剪枝
layer = 5  # attention 评估层
keep_ratio = 0.35  # 保留 top-35% visual tokens

# Prefilling: 在指定层评估 attention
for l in 1..layer:
    Q, K, V = H W_Q^l, H W_K^l, H W_V^l
    H = MHA(Q, K, V) + FFN(H)  # 正常 prefill 到 layer 5

# 在第 5 层计算所有 text tokens 对 visual tokens 的平均 attention
A_layer5 = attention_weights(Q_text, K_visual)  # shape: (N_q, N_visual)
A_avg = mean(A_layer5, dim=0)                   # 对 text tokens 取平均

# 一次性剪枝
keep_idx = topk(A_avg, k=keep_ratio * N_visual)
prune_idx = complement(keep_idx)

# KV cache 永久修改
KV_cache_visual = KV_cache_visual[keep_idx]  # 保留高 attention token
# 被剪枝的 token: 永久丢弃！← 核心问题

# 后续 decoding: 使用固定剪枝后的 KV cache
for t in decoding:
    output_t = LLM_decode(KV_cache)  # 剪枝集不再改变
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FastV (ECCV 2024) 开源实现：github.com/pkunlp-icler/FastV，基于 LLaVA 推理代码，通过分析 attention distribution 决定剪枝层和比例。LLaVA-PruMerge 开源实现：github.com/42Shawn/LLaVA-PruMerge，基于 CLIP attention sparsity。两者均通过减少 KV cache 中 visual token 数量来降低 decoding 阶段的 attention 计算量。DyCoke 实验证明：在相同 FLOPs 下，dynamic pruning (DyCoke) 显著优于 one-shot pruning (FastV, PruMerge)，尤其在 VideoDC、ActivityNet-QA 等需要动态理解的任务上。

涉及论文标题：
- DyCoke__Dynamic_Compression_of_Tokens_for_Fast_Video_Large_Language_Models
