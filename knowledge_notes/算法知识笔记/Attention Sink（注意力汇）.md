## Attention Sink（注意力汇）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sink 是 Guangxuan Xiao et al. (ICLR 2024) 在 StreamingLLM 中首次发现并命名的现象：在 LLM 的自回归解码中，初始几个 token（通常为 system prompt / BOS token）会吸收不成比例的大量注意力分数（attention scores），成为 attention 分布的"汇"（sink）。即使这些初始 token 的语义不重要，删除它们也会导致模型输出质量急剧下降（perplexity 飙升）。Attention sink 之所以产生，是因为 softmax 要求所有注意力权重和为 1，而初始 token 的 Key 经过训练形成了较大的范数，天然吸引注意力。StreamingLLM 利用这一现象，在推理时保留少量 sink tokens（通常 4 个）+ 近期窗口 tokens，实现无限长度流式解码。StreamingVLM 将此概念扩展到多模态视频流领域，将 attention sink（Tsink=512 个初始 text tokens，包括 system prompt 和早期文本）纳入 KV Cache 保留策略，作为稳定长程视频流式推理的三要素之一（sink + text window + vision window）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Sink in Softmax Attention
# Q, K: [seq_len, d_head]
scores = Q @ K.T / sqrt(d_head)  # [seq_len, seq_len]
attn_weights = softmax(scores, dim=-1)

# 现象: attn_weights[:, 0:4] 的值显著大于其他位置
# 原因是前几个 token 的 K 范数大 ||K[:4]|| >> ||K[4:]||

# StreamingLLM 利用 Attention Sink:
# KV Cache 中永久保留 sink tokens + 近期 window tokens
# sink_indices = [0,1,2,3]  # 前 4 个 token 永不驱逐
# 推理时 attend_to = [K_sink | K_recent_window]
```

Annotations: softmax 约束 Σ attn_weights = 1 导致 attention sink 必然发生。Qwen2.5-VL 架构中，StreamingVLM 使用 Tsink=512（远大于纯文本 StreamingLLM 的 4 个），因为多模态场景中 system prompt + vision-related context 更长。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 StreamingVLM 中，attention sink 的使用方式：(1) SFT 训练时，截取前 Tsink 个 previous text tokens + 后 Twindow 个 recent text tokens 作为 SFT sample 的 context；(2) 推理时，Tsink tokens 永久留在 KV cache 中不被驱逐，作为 attention 稳定性的锚点；(3) Ablation 显示 Tsink=512, Twindow=512 是最优配置（Table 5），纯 sink（Twindow=0）仍能保持较高 win rate，纯 window（Tsink=0）则性能下降。Attention sink 对 fine-tuned 模型尤其重要——ReKV 等训练无关驱逐方法破坏 sink token 布局导致输出异常（Table 2）。原始论文：Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024。代码开源：https://github.com/mit-han-lab/streaming-llm。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding

VideoNSA 从 learnable sparse attention 角度分析了 attention sink 行为。论文发现 NSA 的三个稀疏注意力支路表现出截然不同的 sink 模式：(1) Compression 支路产生最多 sink（token merging 放大某些 token norm 同时抑制其他），形成沿 value norm 轴的带状集中分布；(2) Selection 支路几乎无 sink（top-k block filtering 强制平滑 value norm 分布）；(3) Sliding Window 支路在局部邻域边界产生周期性稀疏 sink peaks。通过动态 gating 加权融合三支路后，VideoNSA 整体 sink 比例仅 0.3%，远低于 dense FlashAttention。此外，dense attention 的 sink 均匀分布在全序列，而 VideoNSA 的动态稀疏机制使 sink 在时间轴上更平滑分布，缓解对序列起始位置的过度依赖。这些发现表明 learnable sparse attention 可以通过控制不同支路的 value norm 分布来主动管理 attention sink 的产生和位置分布，而非被动接受或绕过 sink 现象。

公式定义（VideoNSA 沿用 Pai et al., 2025a）：
```
Attention Sink = 1{α > 0.1 ∧ ||v|| < median(||v||) - 2*IQR(||v||)}
```
其中 α 是 key 收到的平均 attention score，||v|| 是 token 的 value norm。

**Visual Attention Sink in MLLMs from VisiPruner**：VisiPruner (EMNLP 2025) 通过系统消融实验首次揭示，MLLM 中的视觉 token 在浅层（layer 1-7）和深层（layer 26+）均表现为 attention sink——即高 attention score 不代表高信息贡献。具体证据：(1) 浅层中最高 attention 的视觉 token 在不同输入指令下保持不变（静态 attention pattern），mask 这些 token 几乎不影响性能（Tab. 1）；(2) 视觉 sink token 的 value vector L1 norm 显著低于非 sink token——因为 softmax 归一化迫使多余的 attention mass 流向 value norm 极小的 token，类似 bias term；(3) 浅层 layer 1 中视觉和文本 token 的 value vector 分布迥异（Fig. 7），需要 modality-specific sink；layer 2+ 文本 token（如 system prompt）即可替代视觉 token 作为 attention sink；(4) 移除浅层视觉 sink token 后，attention mass 会自动重新分配到文本 sink token（App. E.2），进一步证实 sink 的"结构稳定"角色而非信息传递角色。VisiPruner 利用这一发现，在 layer 1 将所有视觉 cross-attention 合并到单个随机 token 作为 attention sink，layer 2+ 完全跳过视觉 attention，保持性能不变。

**VisionSelector 对 Attention Sink 的分析与应用**：VisionSelector 通过端到端可学习的 token 重要性评估，有效规避了 attention sink 对启发式剪枝方法的负面影响。具体机制：(1) 论文假设（Sec 4.3）：attention-based baseline（FastV, VisionZip）在极端压缩率下性能崩塌（VisionZip 从 20%→10% 下降 ~14pp），原因是预训练 attention map 中的 attention sink 偏差——早期位置 token 获得不成比例的高 attention，但语义上不相关——在 10% 极低保留率下迫使模型保留这些"位置优先但语义无关"的 token，导致性能崩溃。(2) LIS 通过自己的 QK^T 全局交互计算重要性得分，而非依赖 MLLM 内部预训练 attention map，因此不受 attention sink 偏差影响（LIS 的注意力矩阵 A = QK^T/√d 在训练中被 CE loss 端到端优化，learned importance 反映语义重要性而非位置偏见）。(3) 可视化（Fig. 3/5/6/7/8）：VisionSelector 的温度图中，关键语义区域（如电话号码、文字、logo）获得高 score，而 background tokens 为低 score；VisionZip 则因 attention sink 保留了位置靠前但无信息的 background tokens。(4) 正向发现（MME 30% 保留率 100.07%）：learned selection 滤除噪声 token 后可实现增益性压缩——这与 VisiPruner 的"移除浅层视觉 sink token 不影响性能"发现一致，表明 VisionSelector 的 LIS 实际上也学会将 attention mass 从视觉 sink token 重定向到语义关键的 token。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
- VideoNSA__Native_Sparse_Attention_Scales_Video_Understanding
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs
