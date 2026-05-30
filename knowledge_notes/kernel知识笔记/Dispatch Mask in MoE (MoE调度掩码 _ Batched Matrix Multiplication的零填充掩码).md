## Dispatch Mask in MoE (MoE调度掩码 / Batched Matrix Multiplication的零填充掩码)

术语是什么？
Dispatch Mask 是 MoE 中 batched matrix multiplication 所需的大型稀疏张量，用于将 token 按 gating 决策重新排列为 per-expert 的连续 batch。其维度为 (num_padded_tokens, num_original_tokens)，是一个二值稀疏矩阵——mask[i,j] = 1 表示 padded token i 对应于 original token j。ES-MoE 指出，该 mask 的内存占用极大：训练 MoE-L batch_size=32, 1024 tokens/batch 时，mask 至少需要 48 GiB 显存。这是 ES-MoE 选择放弃 batched MM、改用 sequential expert computation 的关键动机之一。

从kernel调度角度拆解：
Dispatch Mask 的创建与使用：

```python
# Batched MM 方式（Fairseq, Tutel）:
def batched_moe_forward_with_mask(tokens, gate_output):
    # Step 1: 确定每个 expert 的 token 分配
    expert_token_counts = count_tokens_per_expert(gate_output)
    max_count = max(expert_token_counts)  # 由最热门 expert 决定
    
    # Step 2: 创建 Dispatch Mask
    # Shape: (num_experts * max_count, total_tokens)
    # 每行对应一个 padded expert slot
    dispatch_mask = torch.zeros(
        num_experts * max_count, total_tokens,
        dtype=torch.bool, device='cuda'
    )
    
    # Step 3: 填充 mask
    for token_idx, expert_id in enumerate(gate_output):
        expert_offset = expert_id * max_count
        slot = next_available_slot[expert_id]
        dispatch_mask[expert_offset + slot, token_idx] = True
        next_available_slot[expert_id] += 1
        # 未使用的 slots 保持为 0 → zero-padding
    
    # Step 4: Token 重排 (通过 sparse-dense matmul)
    padded_tokens = dispatch_mask @ tokens  # (E * max_count, d_model)
    
    # Step 5: Batched Expert FFN
    # Reshape + Batched GEMM: (E, max_count, d_model) × (E, d_model, d_ff)
    padded_tokens = padded_tokens.reshape(num_experts, max_count, d_model)
    expert_outputs = torch.bmm(padded_tokens, expert_weights)  # zero-padding 参与计算
    
    return unpermute(expert_outputs)
```

内存消耗详解：`dispatch_mask` 的内存 = `(num_experts * max_count * total_tokens) bits`。MOE-L, batch=32, 1024 tokens/batch, 32 experts → max_count 可能达 few hundred。Mask size = 32 × 300 × 32768 bits ≈ 39 MB... 等等，论文说 "48 GiB"。让我重新看论文：论文说的是 "training MoE-L with a batch size of 32 and 1024 tokens per batch requires at least 48 GiB for the mask"。这应该是指每 device batch size 32 × 1024 tokens per batch = 32768 tokens。如果 `num_experts=128`, `max_count=1024`，则 mask 大小为 `128*1024*32768*4_bytes (int32)` ≈ 17 GB... 具体实现细节论文未完全展开。但核心观点是：dispatch mask 是显存瓶颈，ES-MoE 通过顺序处理避免。

术语一般如何实现？如何使用？
- Fairseq GShard: 通过 `torch.sparse.mm` 或等效的 index-based scatter/gather 实现 token 重排
- Tutel: `sparse_coo_tensor` + 优化的 CUDA kernel 执行 dispatch/combine
- MegaBlocks: 使用 block-sparse matmul 替代传统 dispatch mask，减少 0 值块的内存和计算
- ES-MoE: 完全抛弃 dispatch mask，改用 per-expert 顺序计算——逐个 expert 独立处理其 tokens

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training



术语是什么？
Bit-packed Encoding 是一种利用 Bfloat16 浮点格式中 underutilized exponent bits 存储额外 metadata 的编码技术。PuzzleMoE 观察到 MoE 模型的 expert weights 在 Bfloat16 下 exponent 值集中在 [112, 128] 的窄范围（仅需 5 bits 编码 32 个值），而 Bfloat16 标准分配 8 bits 给 exponent。通过将 exponent 整体减去 112 的 shift 操作（round-up 所有 <112 的值到 112），exponent 可映射到 [0, 31]，释放出 3 bits。这 3 个 bits 用于嵌入：(1) 2 bits 的 binary mask（每个 merged expert pair 中两个 expert 各 1 bit，标记该位置是否属于该 expert）；(2) 1 bit 的 sign（标记原始权重的符号）。

从kernel调度角度拆解术语：
Bfloat16 标准格式：| bit15: sign | bits14-7: exponent (8 bits) | bits6-0: mantissa (7 bits) |
Packed Bfloat16 格式：| bit15: sign_of_expert_1 | bit14: sign_of_expert_0 | bit13: mask_of_expert_1 | bit12: mask_of_expert_0 | bits11-7: shifted_exponent (5 bits) | bits6-0: mantissa |

**Bit-packing 流程：**
1. Input: W_merged（FP32/BF16 merged weight），M_i, M_j, S_i, S_j
2. For each element [p,q]:
   a. Extract raw BF16 fields from |W_merged[p,q]|
   b. exponent ← max(raw_exponent, 112) - 112 → fit in [0, 31]
   c. Pack: packed = (S_i << 15) | (S_j << 14) | (M_i << 13) | (M_j << 12) | (exponent << 7) | mantissa
3. Output: packed_BF16 tensor（视为标准 Bfloat16 tensor，PyTorch 可正常加载，解释为数值时会因 exponent shift 产生偏置，但仅通过自定义 kernel 使用）

**On-the-fly Decoding（CUDA kernel 内）：**
```
mask_bit ← (W ≫ (13 - expert_pos)) & 1
if mask_bit == 0: return 0
sign_bit ← (W ≫ (15 - expert_pos)) & 1
exp ← (W & 0x0F80) + (112 ≪ 7)  // 恢复原始 exponent
W_decoded ← (sign_bit ≪ 15) | exp | (W & 0x007F)
```

术语一般如何实现？如何使用？
- 前提条件：模型权重的 exponent 分布需集中在窄范围（已验证 Mixtral-8x7B, Deepseek-MoE, Qwen1.5-MoE, Qwen3-MoE）。Shift 操作对 perplexity 无影响（Mixtral PPL before=4.37, after=4.37）。
- Packed 数据仍为 16-bit——无需额外 metadata 存储，消除 CSR 等稀疏存储格式的 index overhead。
- 通用性：其他工作如 LEXI（Huffman coding of exponents）、Schrödinger's FP（delta exponent encoding）、Exponent Sharing（LUT-based）也利用了 Bfloat16 exponent 冗余，但 PuzzleMoE 是首个将 freed bits 用于 embedding mask/sign 并配合 custom GPU kernel 做 MoE 推理的。
- 限制：bit 预算受可用 free exponent bits 约束——k=3 合并需 5 bits（3 mask + 2 sign），超出 Bfloat16 的 3 个 free bits，因此不支持 >2-way 合并。

涉及论文标题：
- PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference
