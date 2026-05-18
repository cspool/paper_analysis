## Expert Offloading（专家卸载）

术语是什么？
Expert Offloading是MoE serving中将未被当前token激活的expert权重从GPU显存卸载到CPU memory（或更慢的存储），并在需要时加载回GPU的技术。因为MoE模型每个token只激活top-k个expert（如2/8），72%-84%参数为inactive，将这些inactive expert从GPU移除可显著降低显存占用。但offloading引入了CPU-to-GPU expert loading延迟，形成latency-memory trade-off：缓存更多expert则显存大但miss少，缓存少则显存省但miss时on-demand loading拉高延迟。

从系统架构角度拆解术语：
```
# Expert Offloading生命周期（FineMoE视角）：
1. Expert驻留决策：选择哪些expert保留在GPU cache，哪些offload到CPU
2. Expert Prefetching：在expert被实际使用前，从CPU异步加载到GPU cache
3. On-Demand Loading：expert miss时，暂停普通prefetch立即加载缺失expert
4. Expert Eviction：GPU cache超限时，按eviction priority驱逐expert到CPU
5. Cache管理：GPU侧expert cache容量由GPU显存限制（如6GB）
```
核心trade-off：cache limit越大→expert hit rate越高→on-demand loading延迟越低，但GPU memory footprint更大→留给KV cache和activation的显存更少。

术语一般如何实现？
DeepSpeed-Inference实现expert-agnostic layer-wise offloading（按层on-demand加载，无expert级prefetch）。Mixtral-Offloading用layer-wise speculative prefetching + LRU cache。MoE-Infinity用request-level Expert Activation Matrix + 同步prefetch。ProMoE用stride-based speculative prefetching + per-layer NN predictor。FineMoE用iteration-level Expert Map Store + semantic/trajectory similarity + 异步publisher-subscriber路径实现细粒度offloading。各方法均在HuggingFace Transformers或MoE-Infinity codebase上构建，使用CUDA Runtime APIs做CPU↔GPU数据传输。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
