## PowerAttention (Power-of-2 Sparse Attention / 指数感受野稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PowerAttention 是一种新型静态稀疏注意力模式，核心思想是让每个 token 仅关注 block 距离为 2 的幂次的位置（power-of-2 distances），配合局部滑动窗口和 attention sink tokens。其 mask 定义的核心操作为 `(blk_qk & (blk_qk - 1)) == 0`，即仅保留 block 索引差值为 2 的幂次的注意力连接（差值为 1, 2, 4, 8, 16, 32, ...）。

理论保证（定理 B.1）：在 d 层 LLM 中，每个 token 可访问距离 ≤ 2^d 的所有 token，同时每个 token 的出度 ≤ log n。这同时实现了：(1) 指数级感受野增长（最优扩展效率）；(2) 完整 token 覆盖率（无盲区）；(3) 亚线性出度（高稀疏度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**PowerAttention Mask 构造（Algorithm 1）**：

```python
# q_idx [M, 1]: query token 索引
# kv_idx [1, N]: key-value token 索引
# block_size = 256 (CUDA block size)
# window_size = 5 (sliding window block 数)

# 1. Sink token mask（序列开头的初始 token）
mask_sink = kv_idx < block_size  # [1, N]

# 2. Sliding window mask（局部上下文）
blk_qk = q_idx // block_size - kv_idx // block_size  # [M, N]
mask_window = blk_qk < window_size  # [M, N]

# 3. PowerAttention mask（核心创新）
# 位运算技巧: x & (x-1) == 0 当且仅当 x 是 2 的幂
mask_power = (blk_qk & (blk_qk - 1)) == 0  # [M, N]

# 4. 因果性
causal = q_idx >= kv_idx  # [M, N]

# 5. 组合所有 mask
mask = causal & (mask_window | mask_power | mask_sink)  # [M, N]
```

**PowerAttention 的指数感受野扩展原理（Theorem B.1 路径构造）**：

```
给定 query token i 和 key token j（j < i）:
  距离 d = i - j（在 binary 表示中最多有 log n 个 1）
  设 k₁, k₂, ..., k_m 为 d 二进制中 1 的位置（m = popcount(d)）
  路径: i → (i-2^{k₁}) → (i-2^{k₁}-2^{k₂}) → ... → j
  路径长度 = m ≤ log n

  例如 d = 13 = 0b1101 → k = {0, 2, 3}
  路径: i → (i-1) → (i-1-4) → (i-5-8) = j
```

**配置参数（PowerAttention 论文 4.1 节）**：
- window_size = 5 blocks (5 × 256 = 1280 tokens)
- sink_size = 1 block (256 tokens)
- power blocks ≈ 4 个典型的 power-of-2 位置（取决于序列长度）
- 总计每 token 最多关注 ~10 blocks = 2560 tokens
- 在 32K context (128 blocks) 下稀疏度 ≈ (128-10)/128 ≈ 92%
- 在 128K context (512 blocks) 下稀疏度 ≈ (512-10)/512 ≈ 98%

**时间复杂度**：O(N log^2 N)。每个 query 需要处理的 power-of-2 KV blocks 数为 O(log n)，window blocks 为常数，sink blocks 为常数，总 KV blocks = O(log n)。最终注意力计算量 = N × O(log n) = O(N log^2 N)，接近 sliding window 的 O(N)。

术语一般如何实现？如何使用？

PowerAttention 使用 PyTorch FlexAttention 实现 mask 定义，结合 Triton 进行序列并行训练（RingAttention）。在推理时，mask 预编译为 block-sparse kernel，利用 FlexAttention 自动将 mask 映射到 GPU tiling 策略。训练策略：先在 SlimPajama (1B tokens) 上继续预训练，再用 ChatQA 2 fine-tuning（含跨窗口 long-range dependencies），使模型学会利用指数感受野进行信息检索。

实际应用采用 Hybrid Architecture：每 7 层保留 2 层 Full Attention（保证 sink token 和复杂语义处理），其余 5 层使用 PowerAttention（最大化稀疏效率收益）。在 128K context 下，PowerAttention prefilling 比 Full Attention 快 3.0×，解码仅需 58% 的时间；kernel 开销比 Full Attention 快 21.6×。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

---
