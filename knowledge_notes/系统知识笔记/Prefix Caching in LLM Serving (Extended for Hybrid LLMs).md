## Prefix Caching in LLM Serving (Extended for Hybrid LLMs)

术语是什么？
Prefix Caching是LLM serving系统中的一项优化技术，通过复用之前请求中已计算的模型状态来避免对共享输入前缀的重复prefill计算。在传统纯Attention Transformer serving中基于KV cache的per-token存储实现（可任意切片复用）。但在Hybrid LLMs（Attention + SSM混合架构）中，SSM层使用recurrent state的in-place更新——无法回滚到前缀中间位置，因此Hybrid LLMs的prefix caching需要额外解决SSM state的checkpoint和恢复问题。

Marconi提出面向Hybrid LLMs的prefix caching方案：(1) Judicious Admission——通过radix tree将前缀复用模式分类为Purely Input（多请求共享的系统提示词等）和Input+Output（对话历史续写），仅缓存高复用概率状态，每序列至多2个SSM checkpoint；(2) FLOP-Aware Eviction——淘汰时综合recency和FLOP Efficiency（节省的FLOPs/内存占用）。

从系统架构角度拆解术语：
```
Hybrid LLM Prefix Caching工作流:
  Step 1 - Speculative Admission:
    请求插入radix tree → branching point（intermediate node）→ purely-input → admit
    → leaf node → input+output → 仅缓存最后token的SSM state
  Step 2 - Cache Lookup: radix tree匹配 → Attention KV切片复用 / SSM state恢复 → tail prefill
  Step 3 - Eviction: Utility = recency + α × flop_efficiency → 淘汰最低分
```

术语一般如何实现？如何使用？
Marconi在radix_cache_hybrid.py实现统一radix tree管理KV+SSM states。radix_cache_vllm.py适配vLLM。效果：NVIDIA Mamba2-Hybrid-7B上vs fine-grained checkpointing token hit rate提升4.5×–34.4×，P95 TTFT降低36.1%–71.1%。开源：https://github.com/ruipeterpan/marconi

涉及论文标题：
- Marconi: Prefix Caching for the Era of Hybrid LLMs

---
