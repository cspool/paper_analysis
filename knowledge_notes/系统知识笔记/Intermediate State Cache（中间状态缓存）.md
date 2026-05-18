## Intermediate State Cache（中间状态缓存）

术语是什么？通过联网搜索让回答具体和精准。
Intermediate State Cache是Laser为支持layer-level调度而设计的GPU内存缓存，存储请求在transformer layer边界处的中间hidden states。与KV cache存储key-value向量以重用不同，intermediate cache使得系统可以在layer边界中断请求执行、保存当前状态、并在后续从同一点恢复执行。当layer-level scheduler决定在layer k抢占某个prefill chunk或暂停某个decode请求时，其当前layer的hidden states被写入intermediate cache；恢复时从cache读取并从中断layer继续。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Intermediate cache的抢占-恢复流程：
```
// Req1执行prefill chunk到layer 3，Req2到达需抢占
1. Req1 forward pass through layers 1, 2, 3
2. Scheduler: Req2 slack < remaining iteration time of Req1
3. Executor at layer 3 boundary:
   for each token t in Req1:
       intermediate_cache[req_id=1, layer=3, token=t] = hidden_state[t]
       // hidden_state dim = model_hidden_dim (4096 for Qwen-14B)
   state_manager.mark("Req1", status=PAUSED, resume_layer=3)
4. Req2处理 (可能与Req1合并从layer 3+1继续)
5. Req1恢复时:
   hidden_states = intermediate_cache.load(req_id=1, layer=3)
   Executor.restore(Req1, from_layer=4, with hidden_states)
   // 继续执行layer 4..N，而非从layer 1开始
```

内存管理：intermediate cache参考KV cache的paged memory设计，以fixed-size pages管理。Cache容量：prefill instance 16384 tokens，decode instance 2048 tokens，Llama-70B上消耗<256 MB GPU memory。缓存满时evict最relaxed SLO的请求（因其slack最大，最易恢复），待有空间后异步恢复。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Laser在GPU HBM中分配intermediate cache区域，通过fused CUDA kernel合并state caching和retrieval为单次内存操作以减少roundtrip。KV cache迁移也按layer粒度异步进行并与prefill computation overlap。若动态context length导致memory exhaustion或SLO violation，Laser通过live cache migration重新分配decode请求。256 MB的小footprint使intermediate cache的额外内存开销在80GB A100上可忽略。

涉及论文标题：
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

---
