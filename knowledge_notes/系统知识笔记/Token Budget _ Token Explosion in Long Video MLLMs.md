## Token Budget / Token Explosion in Long Video MLLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Budget（token 预算）是长视频 MLLM 推理中的核心约束——MLLM 对输入 visual token 数量有上限（由 GPU 显存、attention 计算复杂度和 LLM context length 共同决定），而长视频天然产生巨量 token。例如 1h 视频 @ 30fps = 108K 帧，若每帧编码为 196 tokens（典型 ViT），总 visual tokens 达 21M——远超任何 MLLM 的 token 容量。Token Explosion 指随着视频时长线性增长，visual token 数量爆炸式增长，导致推理不可行。FOCUS 的动机陈述：MLLM 的图像→视频扩展中，"aggressive downsampling"（如仅取 64 帧）常因 uniform sampling 错过关键内容，而"increasing frame rate"则直接导致 token explosion。解决方案分为两类：(1) Token compression（减少每帧 token 数，如 DyCoke, HoliTom 的 token merging/pruning）；(2) Keyframe selection（在大量帧中选少量关键帧，如 FOCUS, AKS）。FOCUS 属于第二类，目标是在 strict token budget k（如 k=32/64 帧）下选出最优 k 个 keyframes。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Token Budget 约束在 FOCUS pipeline 中的体现：
```
# 输入: 1h 视频 (108K frames @ 30fps), GPU H100 80GB
# Token Budget: k = 64 frames × 196 tokens/frame = ~12.5K visual tokens

# 若不做 keyframe selection, uniform 64 frames:
#   → 每隔 1688 帧取 1 帧 → 关键事件大概率被跳过

# FOCUS 的工作流:
# Step 1: 在 Token Budget 预算约束下建模为 bandit
#         108K frames → M=225 clip arms (16s each)
# Step 2: Stage I 每 arm 采 q 帧 (共 225q 帧 ≈ 1.0% total)
#         → batch BLIP forward → per-arm relevance stats
# Step 3: Stage II 粗选 arm 采 z 帧 (共 α*m*z 帧 ≈ 0.6% total)
#         → batch BLIP forward → 选 top-m arm
# Step 4: 从 m 个 arm 内选 k=64 keyframes → MLLM 推理
# 总 BLIP forward: 1.6% frames, 5.5 GPU hours
# 总 MLLM input: exactly k=64 frames (符合 token budget)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Token Budget 约束决定了 FOCUS 的所有超参数设计：(1) clip_length=16s 平衡 arm 粒度与采样效率；(2) α=0.25 控制粗选 arm 数，决定 BLIP overhead；(3) k=32/64 由 MLLM 架构固定（Qwen2-VL: 32, LLaVA-Video: 64）。Token Budget 管理是 video MLLM serving 系统的核心问题：vLLM 通过 PagedAttention 管理 KV cache memory budget；DyCoke 通过 TTM + Dynamic Pruning 压缩 visual token budget；FOCUS 通过 bandit keyframe selection 在 token budget 下最大化 frame relevance。三者在不同层面操作同一约束——memory budget, compression budget, selection budget。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding
