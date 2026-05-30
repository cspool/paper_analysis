## PFT (Padding-Free Token Buffers / 无填充Token缓冲区)

术语是什么？

PFT 是 X-MoE 提出的一种稀疏数据结构，用于替代传统 MoE 训练中的 zero-padded expert buffers。传统 GShard 式 pipeline 为每个 expert 分配固定容量 C 的 token buffer [E, C, H]，不足 C 的槽位零填充。PFT 仅存储实际路由到各 expert 的有效 token，通过配套的 ERI-arrays 追踪路由信息。

PFT 结构包含：
- **token_buffer x**：[B, H]，B 为实际路由 token 总数（不含 padding），仅存有效 token
- **ERI-arrays**（Expert Routing Information Arrays）：
  - `token_ids` [B]：每个 token 在原始序列中的位置索引
  - `expert_ids` [B]：每个 token 被路由到的 expert 编号
  - `tokens_per_expert` [E]：每个 expert 接收的 token 数量
  - `combine_weights` [B]：每个 token 的 gating 概率权重

从算法pipeline角度拆解：

PFT 构造和使用流程：

```
# === PFT Construction ===
# Input: top_experts [S, K], combine_weights [S, K], max_token_count
flat_top_experts = flatten(top_experts)  # [S*K]
flat_weights = flatten(combine_weights)  # [S*K]
sorted_idx = argsort(flat_weights)  # 按权重排序以决定drop哪些token

# One-hot + Cumsum 过滤超出容量的token
one_hot = one_hot(sorted_top_experts, num_classes=E)  # [S*K, E]
rank = cumsum(one_hot, axis=0)  # 每expert内的token序号
mask = rank <= max_token_count  # 超出capacity的drop

# 构建ERI-arrays（仅保留有效token）
token_ids = original_ids[mask]  # [B]
expert_ids = flat_experts[mask]  # [B]
combine_weights = flat_weights[mask]  # [B]
tokens_per_expert = histogram(expert_ids, bins=E)  # [E]

# === Padding-free Dispatch ===
# Gather: 按 token_ids 从 gate_out [S,H] gather → dispatch_in [B,H]
# Uneven AlltoAll: 仅传输 B 个有效token（无padding）
dispatch_out = alltoallv(dispatch_in, tokens_per_expert)  # [Bexp, H]

# === Padding-free Combine ===
combine_in = alltoallv(mlp_out, tokens_per_expert)  # [B, H]
# Scatter: 按 token_ids 放回原始位置 + 乘以 combine_weights
output[token_ids[i], :] += combine_in[i, :] * combine_weights[i]
```

优化技巧：PFT construction 中的 cumsum 原为 inner dimension 操作（memory uncoalesced），X-MoE 将 one_hot 转置为 [E, S*K] 在 outer dimension 做 cumsum，加速 10×。

术语一般如何实现？

PFT 需要配套的 kernel 支持：
- **Triton Gather Kernel**：B thread-blocks, 每 block 256 threads，沿 hidden dimension 循环复制，coalesced read
- **Triton Scatter Kernel**：逆向操作 + 加权，coalesced write
- **Sequential GeMM**：按 tokens_per_expert 切片，每 expert 独立 launch GeMM

复杂度：GShard O(ckbsh)+O(ckb²s²) → PFT O(kbsh)

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
