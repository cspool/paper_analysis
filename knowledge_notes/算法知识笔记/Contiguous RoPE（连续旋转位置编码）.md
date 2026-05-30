## Contiguous RoPE（连续旋转位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Contiguous RoPE 是 StreamingVLM 提出的位置编码技术，用于解决流式 KV Cache 驱逐后位置索引断裂和超出训练长度的问题。在标准 RoPE（Rotary Position Embedding）中，每个 token 的位置索引从 0 开始递增，推理时位置可能增长到远超训练最大长度的值，导致 out-of-distribution 退化。Contiguous RoPE 的核心操作是：当旧 token 从 KV cache 中被驱逐后，将剩余和后续 token 的 RoPE 位置索引**左移**，使它们保持与最后保留 token 的**数值连续**。当视频长度超过总窗口尺寸后，有效 RoPE 索引停止增长，保持在有界范围（不超过训练最大长度）。对 Qwen-VL 家族的 3D RoPE（time, height, width），同样应用 contiguous 左移规则。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 标准 RoPE（Native）:
# 推理时位置索引持续增长: pos = [0, 1, 2, ..., T-1]
# 当 T > L_train_max 时，索引超出训练分布

# Contiguous RoPE:
# 维护 position_offset 追踪实际起始位置
# 驱逐旧 token 后，剩余 token 索引左移
position_offset = 0  # 随着驱逐递增
available_positions = list(range(position_offset,
                                  position_offset + cache_len))
# 当 cache_len >= L_train_max 时:
# effective_positions = [position_offset % L_train_max, ...]
# 索引不再增长，在 bounded range 内循环但保持连续性

# 对 3D RoPE (Qwen-VL):
# 3D_indices = build_3d_position(time_idx, h_pos, w_pos)
# contiguous_3d_indices = left_shift(3d_indices, offset)
```

Annotations: Native RoPE 在 infinite stream 上 win rate 从 63.23 降至 25.09（vs GPT-4o, Table 4），Contiguous RoPE 维持 66.18%。100s chunking 可部分恢复 Native RoPE（63.23），但损失长程一致性。Contiguous RoPE 不改变 RoPE 的 base frequency，仅改变 position assignment。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
推理时实现：(1) 在每步推理前计算当前 KV cache 中每个 token 的 contiguous 位置索引；(2) 将位置索引传入模型的标准 RoPE 模块（不修改 RoPE 计算本身）；(3) 对 3D RoPE，需在构建 (t, h, w) 后统一左移 t 维度索引。训练时不需要 contiguous RoPE（训练使用 full attention with native RoPE on short chunks，W=24s 远小于训练最大长度）。适用范围：任何需要在推理时驱逐部分 KV cache、同时保持位置编码连续性的流式 Transformer 推理场景。类比技术：YaRN、LongRoPE（调整 base frequency）、NTK-aware scaling——都解决位置外推问题但策略不同；Contiguous RoPE 关注的是驱逐后连续性而非高频基频扩展。

涉及论文标题：
- StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams
