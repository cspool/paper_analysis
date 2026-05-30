## Star Attention (Two-Phase Block-Sparse Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Star Attention 是一种两阶段 block-sparse 注意力算法（Acharya et al., ICML 2025），专为 Transformer 大语言模型的长序列推理优化设计。它基于一个关键观察：LLM 推理通常包含 (1) prompt encoding 阶段和 (2) token generation 阶段，而许多长上下文任务中，context token 只需 local context、query token 需要 global context。利用这一观察，Star Attention 将注意力计算分解为两阶段：

- **阶段一（Context Encoding）**：将长 context 切分为连续性 blocks，分发到多个 hosts 并行处理。每个 block（除第一个外）前缀拼接 anchor block（第一个 context block），对 2b-token augmented block 做 blockwise-local self-attention。此阶段无跨 host 通信，attention 复杂度 O(L·b) vs full attention O(L²)。多 hosts 完全 embarrassingly parallel。

- **阶段二（Query Encoding & Token Generation）**：Query 被广播到所有 hosts，各 host 使用 Flash Attention 计算 local attention A_h 和 softmax sum s_h，query-host 通过 gather + online softmax（log-sum-exp trick）聚合为 global attention A_global。每 token 仅通信 O(d) 数据（scalar + vector），与 context 长度无关。仅 query-host 更新 KV cache。

Star Attention 兼容几乎所有使用 global attention 训练的 Transformer LLM，无需 fine-tuning。在 RULER benchmark 上，Star Attention 实现 Ring Attention 的 1.1-16.9× 加速，同时保持 97-100% 准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**阶段一：Context Encoding 伪代码**：
```
Require: Context c, Block size b
L ← length(c)
Split c into n = ⌈L/b⌉ blocks: c = [c1, c2, ..., cn]
for i = 2 to n:
    c'_i ← concat(c1, ci)          // prefix anchor block to each context block
end for
Distribute [c'_1, c'_2, ..., c'_n] across H hosts
for each host h concurrently:
    for each assigned block c'_i:
        attn_out = self_attention(c'_i)  // FlashAttention over 2b tokens
        KV = generate_kv_cache(attn_out)
        discard KV for anchor block portion (c1)
        retain KV for ci portion → append to kv_h
    end for
end for
```

**阶段二：Query Encoding & Token Generation 伪代码**：
```
Require: Query q, num output tokens n_o, KV cache kv_h for all hosts
Designate query-host h_q
Broadcast q to all hosts
for t = 1 to n_o:
    for each transformer layer:
        for each host h concurrently:
            A_h = FlashAttention(Q_h, K_h, V_h)     // local attention
            s_h = Σ exp(Q_h K_{h,k}^T / √d)          // softmax denominator
        end for
        Gather all (A_h, s_h) at h_q
        // Online softmax (log-sum-exp) aggregation:
        s_global ← s_1, A_global ← A_1
        for h = 2 to H:
            s_global ← s_global + log(1 + exp(s_h - s_global))
            A_global ← exp(s_h - s_global)·A_global + exp(A_h - s_global)·A_h
        end for
    end for
    next_token = generate(A_global)
    if next_token = EOS: break
    update KV cache at h_q only  // context hosts' KV cache remains frozen
end for
```

**Anchor Block 机制的关键性**：Blockwise-only attention（无 anchor block）会使每个 block 独立产生 attention sink，导致多 sink 分布与 global attention 的单 sink 分布不一致。Star Attention 通过插入 anchor block 使 attention sink 集中在 anchor token 上，丢弃 anchor KV 后分布逼近 global attention。消融实验（Table 4）量化了 anchor block 的作用：无 anchor 时 64K NIAH 准确率 60.1%（vs 99.5% full attention），有 anchor 时恢复至 97.6%。

**Speedup 来源**：加速来自两方面——(a) 阶段一 blockwise-local attention 将复杂度从 O(L²) 降至 O(L·b)，且多 host 并行无通信；(b) 阶段二 distributed softmax 通信量仅 O(d) per token（vs Ring Attention 的 O(L·d) per layer）。当 block size 固定 32K、序列长度从 128K 增长到 1M 时，speedup 从 2.7× 增长到 16.9×。

术语一般如何实现？如何使用？

Star Attention 在 HuggingFace Transformers 和 NVIDIA TRT-LLM 中实现。开源代码：https://github.com/NVIDIA/Star-Attention。使用时关键参数：block_size（建议为序列长度的 1/4，128K 以上固定 32K）、anchor_block_size（建议等于 block_size）。支持 Llama-3.1-8B/70B 和 gradientai 扩展上下文模型（256K/1M）。在 A100 GPU bfloat16 上运行，8B 模型需 8-32 GPU，70B 需 8-32 GPU 取决于序列长度。Flash Attention 被用作阶段一和阶段二的底层 kernel。

涉及论文标题：
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences
