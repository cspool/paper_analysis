## Temporal Redundancy (in Video Understanding / 时间冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Temporal Redundancy（时间冗余）在视频理解中指：视频的相邻帧之间由于拍摄帧率（通常 24-30 FPS）远高于场景变化速率，导致大量连续帧承载几乎相同的视觉信息（如静态背景、缓慢变化的动作）。在 VideoLLM 推理中，每帧被编码为数百个 visual tokens，时间冗余导致大量 visual tokens 承载重复信息，浪费 GPU 显存和计算。RETAKE 将其归类为"低层冗余"（low-level redundancy）——可以仅通过帧间视觉特征距离来检测，无需深层语义理解。传统解决方法包括：(a) 稀疏采样（降低 FPS）——简单但丢失关键瞬时信息；(b) 时序 token 合并（temporal token merging, TTM）——对相邻帧的 visual tokens 做池化或合并，但信息损失不可控；(c) keyframe selection——挑选代表性帧。DPSelect 属于 (c) 并引入峰值感知机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RETAKE 中时间冗余通过 DPSelect 在 visual encoder 输出后减少：
```
# 时间冗余存在于: 连续帧的 visual features M[t] 和 M[t+1]
# 量化: token-averaged cosine distance
d[t] = (1/N) * sum_j (1 - cosine(M[t,j], M[t+1,j]))

# d[t] -> 0: 高时间冗余 (几乎相同的帧)，可安全丢弃
# d[t] -> 1: 低时间冗余 (显著变化)，应保留

# DPSelect 的峰值感知策略:
# - 保留 d[t] 的局部最大值帧 (pivot): 场景切换、动作突发
# - top-k 补充高距离帧: 覆盖渐变过程
# - 丢弃 d[t] 低的帧: 静态场景中的冗余帧

# 与知识冗余的互补关系:
# 时间冗余 -> DPSelect (视觉特征距离, 低计算开销, 信息损失大)
# 知识冗余 -> PivotKV (LLM attention, 有计算开销, 信息保持好)
```
传统 keyframe selection（M2SM, A2Summ, MA-LLM）仅使用帧间距离做 top-N 选择，缺少峰值感知，容易在渐变场景中漏选关键帧。DPSelect 的 max pooling 峰值检测确保保留了每个局部变化窗口中最显著的帧。

T3S 从另一角度利用时间冗余：不试图确定性地选择关键帧，而是通过多次随机帧采样（m 个独立的随机子序列），利用概率覆盖替代精确选择。每次推理随机抽取 N 帧并进一步随机子采样 token（保留率 αᵢ），将 m 个短子序列打包到单次前向传播中并行处理，随后通过 logit 聚合获得最终预测。这一设计的核心论点："随机性是无偏的性质保证"——多试次随机采样在统计上覆盖关键时间片段，无需像学习型选择器那样需要先全量处理所有帧。T3S 同时将 self-attention 复杂度从 O(L²) 降为 O(∑αᵢ²L²)，实现了效率与覆盖的双赢。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DPSelect 在 RETAKE 开源实现（https://github.com/SCZwangxiao/video-ReTaKe）中通过 PyTorch 实现，核心是 torch.nn.functional.cosine_similarity + max_pool1d。参数：window=3（适合大多数视频帧率），alpha_dp 按视频长度自适应（结合 alpha_kv 使 context length <= 32K）。实验验证：DPSelect 在 256 帧限制下性能优于 M2SM、A2Summ、MA-LLM 等 baseline（VideoMME-Long: 51.0 vs. 49.1-50.7）。使用时无需 GPU 之外的额外硬件；时间冗余压缩是最轻量的一步，发生在 LLM 推理之前。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding
