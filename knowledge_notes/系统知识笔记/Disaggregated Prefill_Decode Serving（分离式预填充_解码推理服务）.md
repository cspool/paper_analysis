## Disaggregated Prefill/Decode Serving（分离式预填充/解码推理服务）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Disaggregated Prefill/Decode Serving 是一种 LLM 推理服务架构，将 prefill 阶段和 decode 阶段分配到不同的物理机器（GPU 集群）上执行。传统 co-located serving 中同一 GPU 同时处理 prefill 和 decode 请求，导致两个阶段互相干扰：prefill 是 compute-bound 的大矩阵乘法（GEMM），decode 是 memory-bound 的单 token GEMV 操作，两者的资源需求截然不同。分离式架构将 prefill 机器专门用于输入 token 的并行处理（高计算利用率），decode 机器专门用于逐 token 的自回归生成（可通过大 batch size 将 GEMV 转换为 GEMM 提高利用率）。中间通过高速网络传输 KV Cache，使得 prefill 计算出的 KV Cache 可以发送给 decode 机器使用。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# === Disaggregated Prefill/Decode 请求流程 ===

# Phase 1: Prefill (在 Prefill 机器上)
# 用户请求到达 → Prefill Instance
# 输入: prompt tokens, 序列长度 L_in
H = embedding(tokens)               # (L_in, d_emb)
for block in decoder_blocks:        # 61 blocks for DeepSeek-R1
    H = attention_block(H)          # GEMM 密集型, compute-bound
    H = MoE_block(H)                # compute-bound
KV_Cache = collect_all_layers_KV()  # 收集所有层的 KV Cache

# Phase 2: KV Cache 传输
# 通过高速网络 (NVLink/InfiniBand) 将 KV Cache 
# 从 Prefill 机器传输到 Decode 机器
transfer(KV_Cache, prefill_node → decode_node)

# Phase 3: Decode (在 Decode 机器上)
# 逐 token 自回归生成, 以 batch 形式处理多个请求
for step in range(max_output_tokens):
    token = last_generated_token    # 单 token
    for block in decoder_blocks:
        token_hidden = attention_decode(token, KV_Cache)  # 大 batch GEMV→GEMM
        token_hidden = MoE_decode(token_hidden, experts)
    next_token = sample(token_hidden)
    if next_token == EOS: break
```

核心优势：(1) Prefill 和 Decode 的资源需求不互相干扰——Prefill 机器可充分利用 GPU Tensor Core 做 GEMM，Decode 机器可通过大 batch 提高 ArI；(2) 可以独立扩缩容——当长 prompt 请求增多时扩展 Prefill 实例，当长生成请求增多时扩展 Decode 实例；(3) 降低 Tail Latency——两个阶段不会互相阻塞。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
代表性实现包括 Splitwise (ISCA 2024)、DistServe (OSDI 2024)、WindServe (ISCA 2025) 等。论文中使用 disaggregated 架构作为默认假设，专注于 decode 阶段的优化分析（因 prefill 阶段本身已达到高计算利用率）。当前论文分析的 disaggregated 系统配置：decode 阶段使用 32 B200 GPU 系统（NVLink 5th Gen, 1.8 TB/s），或 32 GPU×8 多实例部署。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
