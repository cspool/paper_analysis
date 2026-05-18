## Inflight Batching（飞行中批处理 / 连续批处理）

术语是什么？

Inflight Batching（也称Continuous Batching）是LLM推理服务中的动态请求批处理技术：当一个decode batch正在GPU上迭代时，新到达的请求可以实时加入当前批处理。与静态批处理（等待所有请求完成后再处理下一批）不同，inflight batching通过暂停正在进行的decode来prefill新请求，然后将新旧请求的decode合并执行，显著提升GPU利用率和系统吞吐量。Orca（OSDI'22）首先提出，现已被vLLM、SGLang等主流框架广泛采用。

从系统架构角度拆解术语：

Inflight Batching在LLM Serving中的运转流程：

1. **当前状态**：Decode batch正在GPU上迭代生成token，包含一组活跃请求。
2. **新请求到达**：新请求携带prompt进入请求队列。
3. **Prefill插入**：调度器暂停当前decode迭代，为新请求执行prefill（计算KV Cache和首token）。在chunked-prefill中prefill chunk与decode iteration融合执行；在MuxWise中prefill在独立SM分区上异步执行。
4. **合并Decode Batch**：prefill完成后，新请求的decode状态合并入当前decode batch，下一轮decode迭代同时处理新老请求。
5. **请求退出**：请求生成EOS或达到max_length时从batch移除，释放KV Cache页。
6. **动态重配**：每轮decode迭代后batch大小动态变化，调度器重新计算资源分配。

关键挑战：inflight batching在prefill期间暂停decode，若prefill过长会导致ITL膨胀。MuxWise通过SM空间分区使decode始终运行，消除这一tradeoff。Bullet进一步通过layer-wise prefill + step-wise decode并发执行，在prefill engine和decode engine各自的SM分区上同时inflight，并通过SLO-aware scheduler动态调整SM分配以平衡prefill和decode吞吐。

术语一般如何实现？如何使用？

在SGLang/vLLM中，inflight batching维护活跃请求池：每次decode迭代后，检查请求队列选择新请求prefill，完成后合并入decode batch。MuxWise中与layer-wise prefill结合：prefill按layer异步发射到独立SM分区，query-based sync检测prefill完成事件后无缝合并，保持inflight batching语义。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
- TetriServe: Efficiently Serving Mixed DiT Workloads

### DiT Serving中的Selective Continuous Batching

在DiT serving中，batching仅在特定场景下有效：(a) DiT是compute-bound的，多步去噪在全量latent tokens上执行，batching的收益主要来自减少kernel launch overhead而非提高GPU occupancy；(b) **仅小分辨率请求受益**：大分辨率请求（1024×1024+）本身已打满GPU，batching反而增加延迟；(c) **需保证SLO**：batching不能使请求超过deadline。TetriServe的selective continuous batching策略：仅对相同小分辨率（256×256, 512×512）的请求在step级别进行batching——如果它们的SLO不会被影响。调度器在round packing阶段判断是否将同一resolution的请求merge到同一batch中执行同一denoising step。这与LLM的inflight batching本质不同：LLM batching针对token级别的prefill/decode混合，DiT batching针对step级别的同分辨率请求合并。

术语是什么？

Token Budget是chunked-prefill技术中的核心调度参数，定义为**单个prefill chunk的新token数与decode batch token数之和**。调度器通过capping token budget来控制prefill+decode融合执行的总计算量，从而保证decode SLO（TBT/ITL）。Token budget同时影响两方面：(1) 若太小，GPU无法饱和，利用率低；(2) 若太大，融合执行的延时超过SLO约束。

从系统架构角度拆解术语：

Token Budget在chunked-prefill调度中的运作：

1. **Budget设定**：在服务启动前，根据SLO目标和模型大小离线tuned token budget。例如Llama-70B在8×A100上，100ms TBT SLO约束下的合规budget约256 tokens。
2. **Prefill Chunk切分**：新请求的prompt被拆分为大小为`budget - decode_batch_size`的chunks（decode batch占用budget的一部分）。
3. **GPU饱和需求 vs SLO约束冲突**（论文§2.3.2）：以Llama-70B 8×A100为例，需要~4K token budget才能打满GPU利用率，但100ms TBT SLO下合规budget仅为~256，相差约16倍。这构成"要么SLO违规、要么利用率低"的两难困境（dilemma between SLO compliance and high utilization）。
4. **Reused Context影响**：chunk prefill需反复读取历史KV cache，当reused context极长时（multi-turn可达50K tokens），即使限制budget也无法满足SLO。

MuxWise通过SM空间分区根本性地绕过token budget dilemma：prefill和decode独立运行在不同SM分区上，decode不受prefill计算量影响，无需budget来约束prefill大小。

术语一般如何实现？如何使用？

在SARATHI-Serve/SGLang中，token budget是一个可配参数（如通过`$CHUNK_SIZE`环境变量），在实验前对每个model-workload对进行离线tuned。调度器根据budget切分prefill chunks，保证每个chunk+decode iteration的总延时在SLO内。MuxWise的contention-tolerant estimator替代了token budget的逻辑：通过预测decode延时和worst-case slowdown，动态确定最优SM分区，而非固定budget。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
