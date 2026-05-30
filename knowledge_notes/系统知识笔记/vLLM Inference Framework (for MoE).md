## vLLM Inference Framework (for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

vLLM 是高性能 LLM 推理服务框架（UC Berkeley, Kwon et al. SOSP 2023），核心创新包括 PagedAttention（将 KV cache 按 page 管理以减少显存碎片）和 continuous batching（动态批处理提升 GPU 利用率）。对 MoE 模型的支持通过 FusedMoE（融合 expert routing + computation kernel）和 Tensor Parallelism/Expert Parallelism 实现。vLLM 支持 HuggingFace 模型直接加载，提供 OpenAI-compatible API server。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

LExI 使用 vLLM 进行 MoE 推理的完整请求流程：

```
1. 模型加载与优化
   vLLM LLM(model="Mixtral-8x7B-Instruct", tensor_parallel_size=4)
   → 加载权重到 4×H100 (TP切分Attention/Norm, Expert分布)
   → LExI 离线优化: k* = LExI_optimize(model, B)
   → 修改每层 top-k: model.moe_layers[j].topk = k_j

2. 请求到达
   Client → POST /v1/completions {"prompt": "...", "max_tokens": 256}

3. Scheduler (Continuous Batching)
   - 将到达的 requests 加入 waiting queue
   - 每个 step: 从 waiting queue 选可调度的 requests
   - Prefill: 新 requests 的 prompt tokens 批量处理
   - Decode: running requests 各生成 1 token
   - Swap: KV cache 不足时 swap 到 CPU memory

4. PagedAttention (KV Cache 管理)
   - 每个 request 的 KV cache 分配为固定大小 page blocks
   - Block table 映射 logical→physical blocks
   - 无需预分配 max_context_length 连续显存
   - Copy-on-write 支持 beam search / parallel sampling

5. MoE Layer Forward (per model layer)
   for each token:
       # FusedMoE kernel
       gate_scores = Router(hidden_state)
       topk_idx, topk_w = TopK(gate_scores, k_j)  // layer-specific k
       # Triton grouped GEMM
       for e in topk_idx:
           expert_out = ExpertFFN[e](hidden_state)
       output = weighted_sum(expert_outputs, topk_w)

6. 响应返回
   vLLM → 流式返回生成的 tokens → Client
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

vLLM 使用 Python/C++/CUDA 实现，核心在 `vllm/core/scheduler.py`（调度器）和 `vllm/worker/model_runner.py`（模型执行）。MoE 支持在 `vllm/model_executor/layers/fused_moe/`。部署命令：`vllm serve <model_name> --tensor-parallel-size 4`。支持多种量化（AWQ/GPTQ/FP8）和分布式推理。LExI 论文中使用 vLLM 的 FusedMoE + Tensor Parallelism 进行 4×H100 上的批量推理（batch_size=16），通过修改每层 top-k 参数实现 layer-adaptive expert allocation。LExI 不修改 vLLM 的调度逻辑、内存管理或 kernel 实现。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference
