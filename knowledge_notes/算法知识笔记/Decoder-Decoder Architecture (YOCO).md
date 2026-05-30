## Decoder-Decoder Architecture (YOCO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Decoder-Decoder Architecture（YOCO，You Only Cache Once）是微软提出的用于 LLM 的新型解码器架构，替代传统 decoder-only Transformer。核心思想是将 L 层均分为两部分：前 L/2 层为 **Self-Decoder**（使用高效自注意力如 gated retention 或 sliding-window attention，仅需 O(1) 常量 KV cache），后 L/2 层为 **Cross-Decoder**（通过 cross-attention 复用 Self-Decoder 最终输出生成的**单一全局 KV cache** K̂, V̂）。该架构的核心优势：(1) KV cache 总量从 O(L×N×D) 降至 O(N×D)，约节省 L 倍 GPU 内存；(2) Prefill 阶段可在 Self-Decoder 完成后提前退出（early exit），因 Cross-Decoder 仅依赖 K̂, V̂，prefill 延迟降至一半以下；(3) Cross-Decoder 的 KV cache 仅需一次 all-gather（分布式训练），而非每层一次。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
YOCO 推理 pipeline（L=26 层，L/2=13，3B model, 512K context）：

```
Input: x[1..N] → Embedding X^0 ∈ R^{N×d}

# === Self-Decoder (layers 1..13) ===
for l in 1..13:
    # Efficient Self-Attention (gated retention, recurrent mode for decode)
    Y^l = ESA(LN(X^{l-1})) + X^{l-1}   # ESA: GatedRetention or SlidingWindowAttn
    X^l = SwiGLU(LN(Y^l)) + Y^l

# === Generate Global KV Cache (once!) ===
M = X^13                              # Self-Decoder's final output
K̂ = LN(M) @ W_K                      # single global key cache
V̂ = LN(M) @ W_V                      # single global value cache

# === Prefill Early Exit: STOP HERE during prefill! ===

# === Cross-Decoder (layers 14..26) ===
for l in 14..26:
    Q̂^l = LN(X^{l-1}) @ W_Q^l        # per-layer query projection
    Y^l = CrossAttention(Q̂^l, K̂, V̂) + X^{l-1}  # all layers share K̂, V̂
    X^l = SwiGLU(LN(Y^l)) + Y^l

# Output
logits = softmax(X^26 @ W_lm_head)
```

**Annotations**: Prefill 时仅执行 Self-Decoder 的前向（13 层而非 26 层），然后生成 K̂,V̂ 并退出。Decode 时 Self-Decoder 用 recurrent 模式（O(1) state），Cross-Decoder 标准 attention 复用 K̂,V̂。KV cache 存储：仅 K̂,V̂（单层 N×d×2）+ Self-Decoder 的常量状态（如 gated retention 的 S ∈ R^{d×d}）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
YOCO 适用于以下场景：(1) 长上下文 LLM 推理——KV cache 内存从 O(LND) 降至 O(ND)，使 1M token 上下文可在 consumer GPU 上部署（3B: 12.4GB）；(2) 低延迟 prefill——early exit 机制将 512K prefill 从 180s 降至 <6s；(3) 分布式长序列训练——Chunk Parallelism 减少通信频率。实现代码开源：https://aka.ms/YOCO。Self-decoder 可选用不同的高效 attention 模块（gated retention 或 sliding-window attention），Cross-decoder 兼容 GQA 进一步节省 KV cache。架构与标准 Transformer 训练流程兼容（使用相同的 AdamW 优化器、SwiGLU FFN、RMSNorm 等），可从标准 Transformer 的预训练设置迁移。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
