## DuoAttention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

DuoAttention（Xiao et al., 2025, MIT & NVIDIA）是一个将 LLM 的 attention head 分类为 Retrieval Heads 和 Streaming Heads 的 KV cache 压缩框架。核心观察：仅少数 attention head（Retrieval Heads）对长上下文处理至关重要，需要 full attention across all tokens；多数 head（Streaming Heads）主要关注 attention sink（首 token）和最近 token，可以使用 constant-length KV cache（仅保留 sink + recent tokens）。

DuoAttention 包含三个阶段：
1. **Retrieval Head Identification（训练）**：为每个 KV head 分配可训练 gate value α_{i,j} ∈ [0,1]，前向 pass 中混合 full attention 和 streaming attention 输出：attn = α·full_attn + (1-α)·streaming_attn。在合成 passkey retrieval 数据集（BookSum + 10×32-word passkeys）上以 L2 distillation loss（最后 hidden state 偏差）+ L1 regularization（λ=0.05）优化 gate values。仅数千可训参数，所有模型权重冻结，2,000 steps on 8×A100 完成。
2. **Head Binarization & Reordering**：按 sparsity quantile threshold τ 将 α 二值化为 {retrieval, streaming}，高于 τ 为 retrieval head。预处理时重排 Q/K/V 投影的输出通道，将两类 head 分为连续簇，推理时用 slicing/concat 替代 scatter/gather。
3. **Deployment（双 KV Cache）**：每层两个 KV cache——retrieval heads 使用 full KV cache（all tokens），streaming heads 使用 constant KV cache（仅 sink 64 + recent 256 tokens）。Chunked pre-filling 中 streaming heads 每 chunk 后立即 prune KV 仅保留 sink+recent，下一 chunk 仅 attend 到 constant 数量的历史 token，pre-filling 复杂度从 O(L²) 降至 O(LK)。

与完全依赖 attention score profiling 的方法（FastGen, RazorAttention）不同，DuoAttention 直接测量 output deviation（压缩 KV cache 后的输出偏差），可以识别 attention scores 中并不明显、但对 long-context 至关重要的 retrieval head。

从算法pipeline角度拆解术语。

**Phase 1: Gate Value Training**
```
# 初始化：α_{i,j} = 1.0 (所有 head 初始假设为 retrieval)
# 合成数据：BookSum + 10×32-word passkeys, 50 长度区间 (1K→max_len)

for step in 1..2000:
    # 前向 (per KV head j in layer i)
    full_out = softmax(Q @ K^T ⊙ M_causal) @ V
    stream_out = softmax(Q @ K^T ⊙ M_streaming) @ V  # Λ-like mask
    attn_{i,j} = α_{i,j} · full_out + (1-α_{i,j}) · stream_out

    # Loss (仅最后 l 个 passkey tokens)
    L_distill = (1/N) Σ (H_full_last - H_mixed_last)²  # L2 on hidden states
    L_reg = Σ_i Σ_j |α_{i,j}|                            # L1 sparsity
    L = L_distill + 0.05 · L_reg

    # AdamW: lr=0.02 warmup(400 steps 0.002→0.02)→decay(400 steps 0.02→0.002)
    # 仅更新 α_{i,j}，模型权重冻结
```

**Phase 2: Binarization & Head Reordering**
```
# 按 sparsity quantile 确定阈值 τ
for each head (i,j):
    type = "retrieval" if α_{i,j} > τ else "streaming"

# Head reordering: 重排 W_Q, W_K, W_V 的输出通道
# retrieval heads → 连续簇 0..R-1, streaming heads → 连续簇 R..H-1
```

**Phase 3: Dual KV Cache Decoding**
```
# Per layer forward:
Q_ret, Q_str = split(Q, head_dim)        # 沿 head 维度切分
K_ret, V_ret = full_kv_cache              # 全量历史 (retrieval heads)
K_str, V_str = sink_and_recent_kv_cache   # 仅 sink + recent (streaming heads)

out_ret = FlashAttention(Q_ret, K_ret, V_ret)
out_str = FlashAttention(Q_str, K_str, V_str)
output = concat([out_ret, out_str], head_dim) @ W_O
```

**Phase 4: Chunked Pre-filling (streaming heads)**
```
for each chunk of K tokens:
    K_chunk, V_chunk = compute_KV(chunk)
    # streaming heads: 立即 prune，仅保留 sink + recent
    K_str = prune_to_sink_and_recent(K_str, K_chunk)
    V_str = prune_to_sink_and_recent(V_str, V_chunk)
# Streaming heads 复杂度: time O(LK) [vs O(L²)], memory O(K) [vs O(L)]
```

**配置与性能**：
| 模型 | Attention | Retrieval Ratio | Memory Reduction | Latency Reduction (Decode/Pre-fill) |
|------|-----------|----------------|------------------|-------------------------------------|
| Llama-2-7B | MHA (32 heads) | 25% | up to 2.55× | 2.18× / 1.73× |
| Llama-3-8B | GQA (8 KV heads) | 50% | up to 1.67× | 1.50× / 1.63× |

GQA 模型的 retrieval ratio 更高（50% vs 25%），因为 GQA 中 per-group gate value 绑定多个 query head，必须保守压缩。MHA 中每个 head 独立 gate，可更激进压缩。

**结合量化**：DuoAttention + QServe (W8A8KV4) → Llama-3-8B 单 A100 容纳 3.3M tokens（6.4× capacity vs full attention BF16, 仅需 0.52 GB per token）。

术语一般如何实现？如何使用？

开源：https://github.com/mit-han-lab/duo-attention。基于 PyTorch + FlashInfer (RoPE/RMSNorm kernels) + FlashAttention-2。训练用 FSDP2 + DeepSpeed Ulysses sequence parallelism 支持长序列。Deployment 默认 sink=64, recent=256, pre-fill chunk=32K。NIAH 上 25% retrieval ratio (MHA) / 50% (GQA) 即可保持 full attention 级别准确率，LongBench 上同样保持接近 full attention 的性能。所有 baseline（H2O, TOVA, StreamingLLM, FastGen）在 NIAH 上均失败（无法在不同深度正确检索）。

与 PruLong 对比：DuoAttention 使用 L2 reconstruction loss + synthetic passkey + continuous gating（有 train-test rounding gap）；PruLong 使用 NTP loss + Hard Concrete Bernoulli + natural long-context data，在 natural data recall 上更优（91.4 vs 38.6 at 70% sparsity）。

涉及论文标题：
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
