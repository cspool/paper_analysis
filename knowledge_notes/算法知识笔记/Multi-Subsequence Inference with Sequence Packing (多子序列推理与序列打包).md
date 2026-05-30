## Multi-Subsequence Inference with Sequence Packing (多子序列推理与序列打包)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Subsequence Inference 是 T3S 提出的推理范式：将 m 个独立采样的短视频子序列打包到一个 batch 中，通过单次 MLLM 前向传播完成推理。每一子序列包含随机采样的 N 帧视觉 token + 文本 prompt token。核心技术是 **Sequence Packing** + **Block-Diagonal Attention Mask**：各子序列在批次维度拼接，但 self-attention 计算时使用块对角线 mask——每个子序列仅与自身 token 交互，不跨子序列关注。这使得每个 attention 块的实际计算量由其子序列长度（αᵢL）决定，而非所有子序列总长度，从而将总复杂度从 O(L²) 降为 O(∑αᵢ²L²)。在 GPU 上的实际效果：打包后总 token 数 = (α₁+α₂)L < L，且每个 attention 块更小，KV cache 占用更少。与传统的 batching（多个请求独立前向）不同，T3S 的 packing 将多个子序列合并到同一 batch 中共享一次前向传播，避免了多次 kernel launch 开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === Multi-Subsequence Inference with Sequence Packing ===

# 假设 m=2, L=1000 (原始视觉token数)
# α₁=0.5 → |v̂^(1)| = 500, α₂=0.3 → |v̂^(2)| = 300
# 文本 tokens |t| = 50

# 打包方案1: 将纹理子序列拼接
# Packed sequence: [v̂^(1)_1, ..., v̂^(1)_500, t_1, ..., t_50, 
#                    v̂^(2)_1, ..., v̂^(2)_300, t_1, ..., t_50]
# 总长度 = 500+50+300+50 = 900

# Block-Diagonal Attention Mask 构造:
# mask[i][j] = 
#     1  if i,j 属于同一子序列-文本块 (0 <= i,j < 550 或 550 <= i,j < 900)
#     0  otherwise (跨块不关注)
#
# 效果: QK^T 计算中，block 1 矩阵 [550×550], block 2 矩阵 [350×350]
#       总参数量 = 550² + 350², 而非 900² = 810000
#       实际节省: 1 - (302500+122500)/810000 = 47.5%

# 打包方案2: 文本tokens共享 (论文实际做法)
# 每个子序列独立拼接文本, 等价于独立推理, 
# 但通过 batch 维度并行处理
seq_1 = concat(v̂^(1), t)  # [500+50]
seq_2 = concat(v̂^(2), t)  # [300+50]
# Batch forward: MLLM([seq_1, seq_2], attn_mask=BlockDiagonal([550, 350]))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
T3S 开源实现使用 PyTorch 的 block-diagonal attention mask 实现。关键超参：m 通常设为 2（m=3/4 收益递减且计算成本递增），αᵢ 的选择需满足 ∑αᵢ² < 1 以获得理论加速。使用方式：在推理时配置 m/N/αᵢ/k 参数，调用 T3S wrapper 的 forward 方法，替代原有的单序列推理。注意：当 α₁+α₂ ≥ 1.3 时，打包后的长序列可能导致 OOM，此时需回退到两个串行前向传播。局限性：(1) 单 GPU 上序列间无真正的并行——chunk 计算已使 GPU 饱和；(2) 多 GPU 设置下可将各子序列分配到不同设备独立推理。

涉及论文标题：
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding
