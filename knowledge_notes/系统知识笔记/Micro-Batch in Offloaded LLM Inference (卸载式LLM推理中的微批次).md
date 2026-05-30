## Micro-Batch in Offloaded LLM Inference (卸载式LLM推理中的微批次)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 GPU 内存受限的 LLM 卸载推理场景中，Micro-Batch（微批次）指一次 GPU kernel 执行所处理的 token 子集大小，记为 μ。由于 GPU memory 不足以容纳整批所有 token 的中间激活，batch size N 被拆分为多个微批次（N/μ 个），每个微批次逐次在 GPU 上执行。Micro-batch 的大小直接影响：(1) GPU peak memory 占用（μ 越大 → 激活内存越大 → 可用于 weight buffer 的空间越小）；(2) operational intensity（μ 越大 → FFN GEMM 的 I 越高 → GPU 利用率越高）；(3) weight transfer amortization（μ 越大 → 单次 weight 加载摊销到更多 token → I/O overhead 占比越小）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 MoE-Lightning 中，μ 是 HRM policy optimizer 搜索的 6 元组策略的核心参数。搜索目标：在 GPU memory 约束下最大化 per-layer throughput。μ 的选择权衡：
- 更大的 μ → 更高的 GPU GEMM 利用率（MoE FFN 为 memory-bound，μ ↑ 增加 arithmetic intensity）→ 更高 compute efficiency
- 更大的 μ → 更高的 GPU peak memory → 更少的 weight buffer 空间 → 可能需要降低 r_w（GPU static weights ratio）
- 更大的 μ → 更少的微批次数 n_ub → 更少的 pipeline stages → 更多 pipeline bubbles

例如：Mixtral 8x7B on T4 16GB，HRM 搜索策略为 μ=36, N=504, n_ub=14——即在 16GB GPU HBM 中，每微批次 36 个 token 的激活 + 2 页 weights 的 buffer 刚好不超限。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- FlexGen 通过 offline search 确定 μ（基于 data fitting）；MoE-Lightning 通过 MILP-based HRM 确定 μ（基于 analytical model）。
- 实践中 μ 受 GPU HBM capacity、model hidden dim、prompt length、data type 等因素影响。
- 在线推理时，通过 Algorithm 2（Request Batching）将变长请求按 descending input length 贪婪分配到各微批次，使每微批次 token 数接近目标 μ。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
