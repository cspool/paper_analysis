## Adaptive Token-wise KV Caching

术语是什么？通过联网搜索让回答具体和精准。

Adaptive Token-wise KV Caching是eLLM系统中提出的LLM推理KV cache管理策略。核心思想是：在decode阶段，每个请求不再缓存所有历史token的完整KV cache，而是只缓存较新的(1-r)比例token的KV（r为uncached token ratio），将较早token的KV cache释放以节省GPU显存。被释放token的KV在后续decode需要时通过GPU SM重算或从host memory swap恢复，用完即释放。这种"compute换memory"策略利用GPU decode阶段SM utilization低、显存紧张的资源不平衡特点，将原本被KV cache占用的显存释放出来容纳更多并发请求，提高batch size和吞吐。eLLM将r作为在线优化变量，在GPU显存约束和TPOT SLO下求解最优值。Adaptive KV Caching与粗粒度all-or-nothing preemption策略（如vLLM-Recompute丢弃整请求、vLLM-Swap整体换出）的根本区别在于细粒度选择哪些token缓存、哪些释放。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Adaptive Token-wise KV Caching在eLLM系统架构中的运转流程（以Llama2-13B在A100上decode为例）：

1. **决策阶段**：eLLM Scheduler启动时，request-level optimizer收集当前所有waiting/running请求的sequence length、队列大小、最大等待时间等metadata，用SciPy SLSQP求解约束优化问题：maximize throughput = b / (compute_time(b, r) + overhead)，subject to GPU memory ≤ M_GPU + M_saved(r) - M_overhead(b, r) 且 predicted TPOT ≤ SLO。输出最优batch size b和uncached token ratio r。

2. **Token划分**：对batch中每个请求，根据r划分历史token。若r=0.4，前40%历史token标记为uncached（KV可释放或swap），后60%标记为cached（KV保留在GPU）。每个SequenceGroup维护自己的r值，不同请求可有不同r（受各自序列长度和历史影响）。

3. **Layer级执行**：进入transformer layer i时，scheduler对cached token直接使用GPU中KV，对uncached token选择recompute（在GPU上重算该层KV投影）或swap（从host memory加载之前保存的KV）。使用后临时KV立即释放。

4. **显存效果**：以Llama2-13B为例，若avg sequence length=1,568 (ShareGPT)，r=0.4则每请求释放约40% KV cache = ~0.4 × 1568 × 40 layers × 40 heads × 128 dim × 2 (K+V) × 2 bytes ≈ 160MB。batch中多请求累积释放的显存可用于增加batch size。论文在L-Eval长上下文上r平均约0.47，memory saving超47%，throughput最高提升3.03×。

5. **Closed-loop反馈**：layer-level执行后计算实际overlap/fusion额外显存Mo，反馈给request-level optimizer。若Mo比预期大则下一轮可能减小r（缓存更多）或减小b（少并发）以保证SLO；若Mo比预期小则可增大b或增大r进一步释放显存。

与baseline的对比：
- vs vLLM-Recompute：不是丢弃后全量重算，而是细粒度选择r比例释放+逐layer按需重算，重算量仅O(r × L × T)而非O(L × T)
- vs vLLM-Swap：不是整请求KV搬移，而是通过uncached token ratio控制仅部分token需要host-GPU传输，其余保留GPU resident
- vs HCache：不是仅加速恢复速度，而是从源头降低每个active request的KV residency（KV常驻占用）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式（基于eLLM论文）：基于vLLM扩展，约3,500行Python + 1,700行CUDA代码。关键实现点：
1. 每个SequenceGroup维护uncached token ratio r，控制GPU/CPU KV block分配比例
2. KV block按F个连续layer重塑为layer-granular block（默认F=4），支持per-layer KV release/recompute的细粒度管理
3. map table维护per-token、per-layer的缓存状态（seq_id, token_id, layer_id, logical_block_id, physical_block_id, #filled）
4. 双CUDA stream实现host-GPU KV传输与GPU重算并行
5. 预编译多组CUDA .so（32-1024线程, step 32）支持不同workload下的fused kernel配置

截至分析时eLLM未公开官方GitHub仓库。

涉及论文标题：
- High Throughput and Low Latency LLM Serving via Adaptive KV Caching
