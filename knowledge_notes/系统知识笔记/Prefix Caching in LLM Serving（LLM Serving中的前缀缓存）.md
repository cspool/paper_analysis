## Prefix Caching in LLM Serving（LLM Serving中的前缀缓存）

术语是什么？通过联网搜索让回答具体和精准。

Prefix Caching是LLM serving系统中的一项优化技术，通过复用之前请求中已计算的Key-Value (KV) cache来避免对共享输入前缀的重复prefill计算。当多个LLM请求共享相同的输入前缀（如system prompt、few-shot examples、多轮对话历史）时，prefix caching存储并复用这些共享前缀的attention states，跳过prefill阶段的重复计算。在vLLM中基于PagedAttention的block-level KV cache管理实现。对agent workload（多轮迭代、长共享prompt前缀）特别有效。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Prefix Caching在AI agent serving中的工作流：

```
// Agent迭代中的prefix caching
迭代1: Prompt = [Instruction(1K), FewShot(500), Query(20)]
       → LLM call #1: prefill计算全部~1520 tokens的KV cache → 存入cache

迭代2: Prompt = [Instruction, FewShot, Query, LLM_History(200), Tool_History(800)]
       → shared prefix: [Instruction, FewShot, Query] ~1520 tokens
       → prefill只计算新增~1000 tokens (history)
       → shared tokens KV cache从cache读取
       → prefill latency降低60.1% (论文测量)
```

论文量化prefix caching对agent workload的系统级影响：
- Prefill latency平均降低60.1%
- End-to-end LLM inference latency平均降低15.7%（agent workload，CoT因decode主导收益较小）
- Serving throughput平均提升5.62×（agent workload vs ShareGPT仅1.03×）
- KV cache memory serving场景下平均降低51.7%、最大降低63.5%
- LATS memory requirement平均降低64.8%（并行LLM calls共享prefix）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

vLLM中通过PagedAttention的block-level管理实现：(1) input token sequence分割为固定大小blocks；(2) blocks hash作为cache key存储；(3) 新请求到达时scheduler检查prefix block hash是否命中；(4) 命中则跳过prefill直接从cache读取KV cache；(5) 未命中正常计算并存储。在token-level scheduler中，prefix caching通过缩短prefill间接提升throughput——长prefill阻塞decode scheduling，缩短prefill减少queue interference，这在agent workload中尤为关键。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
