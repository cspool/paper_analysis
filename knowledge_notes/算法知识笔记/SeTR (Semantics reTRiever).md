## SeTR (Semantics reTRiever)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SeTR (Semantics reTRiever) 是 HERMES 框架中与 ECO 互补的语义信息提取模块，受人类 semantic memory（语义记忆）认知机制启发。SeTR 的目标是从全视频帧特征中提取高层次的语义线索，而非保留每帧的时序细节。算法流程：(1) 归一化全部帧特征 $F \in \mathbb{R}^{B \times N \times T \times C}$；(2) 以 stride=k 将 N 帧分为两组：保留组 K（每 k 帧取 1 帧，得 N/k 帧）和压缩组 K̄（剩余 N-N/k 帧）；(3) 对每个 K̄ 中的帧，计算其与所有 K 帧之间的 dot-product similarity 分数；(4) 将每个 K̄ 帧按元素级平均合并到最相似的 K 帧中。最终保留 $\frac{N}{k}$ 帧作为 semantic representations。在 HERMES 中默认 keep_ratio=0.2（k=5），即从 100 帧压缩到 20 帧语义代表。SeTR 区别于 ToMe（在 ViT 内部层间合并 token）：SeTR 在帧级别（而非 token 级别）操作，保留语义最丰富、最具代表性的帧而非简单的 token 合并。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SeTR 伪代码：
```
F = concat(W_1, ..., W_{N/w})                     # 全部 window features: (B, N, T, C)
F = normalize(F)                                  # 归一化
K_indices = [0, k, 2k, ...]                       # 保留组: N/k 帧
K_bar_indices = rest                              # 压缩组: N - N/k 帧
F_K = F[:, K_indices, :, :]
F_Kbar = F[:, K_bar_indices, :, :]
for each frame f in F_Kbar:
    sim_scores = dot_product(f, F_K)              # 与每个保留帧的相似度
    j* = argmax(sim_scores)
    F_K[j*] = (F_K[j*] + f) / 2                  # 合并到最相似的保留帧
F_prime = F_K                                     # (B, N/k, T, C)
# 后续: Hierarchical Q-Former (fQFormer → vQFormer)
```
SeTR 后的 Hierarchical Q-Former：fQFormer 独立增强每帧语义 → Frame-to-Sequence Adapter (Linear) → vQFormer 全局聚合 → $Q_{sem}$。消融实验（Table 7）：移除 SeTR 导致 accuracy 下降 5%（78.6 → 73.3）；MaxPool/AvgPool 替代 SeTR 分别降至 70.4/73.3；K-Means 聚类压缩为 75.7，均低于 SeTR 的 78.6。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SeTR 的 keep_ratio 是唯一超参数：20% 在 MovieChat-1k 和 Breakfast 上均为最优（Figure 5），验证 HERMES 对超参数鲁棒。SeTR 可独立作为 plugin 插入现有 VLM：(1) MA-LMM + SeTR → accuracy +3.8%, latency 仅 +1.5%（Table 5）；(2) LongVA + SeTR → accuracy +0.45%, latency -27%（Table 3）；(3) LLaVA-OneVision + SeTR → accuracy +1.04%, latency -33%（Table 4）。SeTR 与 ECO 互补：ECO 提供 episode-level temporal detail，SeTR 提供 global semantic themes——将两者 concat 后送入 LLM 实现双流理解。training-free，零额外训练。

涉及论文标题：
- Bridging_Episodes_and_Semantics__A_Novel_Framework_for_Long-Form_Video_Understanding
