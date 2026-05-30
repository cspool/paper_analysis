## Chunked Prefill (分块预填充)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunked Prefill（分块预填充）是一种 LLM 推理调度技术，最初由 Sarathi (2023) / Sarathi-Serve (OSDI 2024) 提出，将长 prefill 请求拆分为多个固定大小的 token chunks，在 decode iteration 之间交错执行，以避免长 prefill 阻塞 decode 导致的 TTFT（Time To First Token）尖峰和 generation stall。在 RETAKE 中，chunked prefill 被用于 VideoLLM 的长视频处理场景：视频被划分为等长的 frame chunks（每个 chunk 包含 tau 帧），逐 chunk 进行 prefilling，每个 chunk 处理完后立即执行 PivotKV 压缩以控制 KV cache 增长。由于 Transformer 的自注意力是因果的（causal），逐 chunk prefilling 在数学上等价于一次性 prefilling 整个序列。Chunked prefill 还在 Sarathi-Serve, vLLM, SGLang 等多个 serving 系统中被广泛采用，用于提升多请求批处理的 GPU 利用率并降低 tail latency。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 RETAKE 的 VideoLLM 推理 pipeline 中，chunked prefill 的执行流程：
```
输入: 视频序列 H (包含 alpha_dp*T 帧的 visual tokens + L 个 prompt tokens)
chunk_size = tau 帧 (每帧 N 个 visual tokens)

# 将 H 划分为 chunks
chunks = [H_1, H_2, ..., H_{alpha_dp*T/tau + 1}]
# 前 alpha_dp*T/tau 个 chunk 各含 tau*N 个 visual tokens
# 最后 1 个 chunk 含 L 个文本 prompt tokens

KV = []  # 初始化空 KV cache

for each chunk H_i:
    KV_i = LLM.prefill_chunk(H_i, KV)  # 计算当前 chunk 的 KV
    
    if H_i 是视频 chunk:
        KV_i = PivotKV_compress(KV_i)  # 压缩视频 chunk 的 KV cache
    
    KV = Concat(KV, KV_i)  # 更新历史 KV cache

# 所有 chunks 处理完毕, KV cache 包含完整上下文
# 随后进行标准自回归 decoding
output = LLM.decode(KV)
```

与标准 serving 中 chunked prefill 的区别：RETAKE 在每次 chunk prefilling 后插入 PivotKV 压缩步骤，而在标准 serving（如 vLLM）中 chunks 之间不修改 KV cache。效率优化：RETAKE 使用额外 CUDA stream 将第 l 层的 PivotKV 压缩与第 l+1 层的 chunk prefilling 重叠执行，将额外开销从 +28%/62% TTFT 降至 +8%/11%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Chunked prefill 在现代 LLM serving 系统中的实现：
- **Sarathi-Serve** (OSDI 2024): 首个系统化提出 chunked prefill 的框架，设定 token budget per iteration，将 prefill chunk 与 decode 请求混合批处理（"piggybacking decodes with chunked prefills"）。开源: https://github.com/microsoft/sarathi-serve
- **vLLM**: 通过 `--max_num_batched_tokens` 和 `--max_num_seqs` 参数控制 chunk size。PR #3121 集成了 chunked prefill 支持。
- **SGLang**: 通过 RadixAttention 实现 token 级 KV cache 复用，内置 chunked prefill 支持（`--chunked-prefill-size` 参数）。

RETAKE 中的 chunked prefill 是论文自实现的（非依赖外部 serving 框架），针对 VideoLLM 场景优化：chunk 按帧（而非 token）划分，每个视频 chunk 后自动触发 PivotKV 压缩。Chunk size tau 的选择需权衡：太小则 PivotKV 压缩频率高（开销大），太大则单 chunk KV cache 峰值高（显存压力大）。论文未明确给出 tau 的最优值。实现代码见 https://github.com/SCZwangxiao/video-ReTaKe。

涉及论文标题：
- ReTaKe__Reducing_Temporal_and_Knowledge_Redundancy_for_Long_Video_Understanding
