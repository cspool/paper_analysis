## Chunked Prefill（分块预填充）

术语是什么？

Chunked Prefill是一种LLM推理调度技术，将长Prefill请求拆分为多个较小的token块（Chunk），在Decode迭代之间交错执行，以避免长Prefill阻塞Decode导致ITL SLO违规。这是PDM之前的主流方案，vLLM和SGLang均支持。

从系统架构角度拆解术语：

调度流程：
1. 将新到达的Prefill请求按token数切分为固定大小的Chunk（如512 tokens）。
2. 每轮Decode迭代前，先执行一个Prefill Chunk的计算。
3. Prefill Chunk完成后，立即执行Decode迭代。
4. 重复上述交替过程，直到所有Prefill请求处理完毕。

Chunked Prefill的问题：Prefill和Decode共享同一组SM，Prefill Chunk执行时Decode完全暂停，导致ITL出现spike；Chunk过大会导致长ITL违规，Chunk过小会导致Prefill效率低。PDM通过SM空间分区解决了这一"要么SLO违规、要么利用率低"的两难困境。

**Token Budget Dilemma量化**（MuxWise §2.3.2）：以Llama-70B部署在8×A100为例，需要约**4K token budget**才能打满GPU利用率；但100ms TBT SLO限制下合规budget仅约**256**（相差约16倍）。此外，chunk prefill需反复读取历史KV cache，当reused context极长时（multi-turn可达50K tokens），TBT随reused长度线性膨胀，极端情况下即使限制budget也无法满足SLO。

**KV Cache Reload Overhead量化**（Bullet §2）：Bullet进一步量化了chunked prefill的KV cache重复读取代价。16k-token prefill以1k chunk切分为16个chunk，形成近似N(N+1)/2次KV reload。实测1k chunk下compute efficiency从71%降至61%，最后一个chunk处理时间为第一个的1.9x，总prefill latency比unchunked高1.13x。2k chunk虽缓解利用率下降，但平均per-chunk latency增加1.86x。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

Laser进一步提出**layer-level chunked prefill**，将chunked prefill的粒度从完整的iteration细化到layer边界：(1) 可在layer边界抢占当前prefill chunk以优先处理latency-critical请求（解决head-of-line blocking）；(2) 可在同一iteration内的layer边界动态合并新请求到当前chunk以形成更大有效chunk（解决小chunk GPU低效问题）。Laser的prefill Scheduler在每层后评估新请求的TTFT slack，若当前chunk剩余时间危及新请求SLO则保存intermediate state并抢占/合并。layer-level粒度使抢占延迟从完整iteration（数十/数百ms）降至单层（<1ms），chunk动态合并也使prefill per-token latency降低45.4%而per-iteration latency仅适度增长。

TokenFlow将SGLang chunked prefill作为baseline进行对比：chunked prefill虽改善prefill对decode的阻塞，但不解决burst场景下buffer不感知调度导致的head-of-line blocking和token过度生成问题。

---
