# 6 Related Work

KV-cache management. PagedAttention [\[22\]](#page-13-11) observes the huge and growing KV-cache memory in LLM inference. Meanwhile, the KV-cache storing may introduce fragmentation and redundant duplication in GPU memory. PagedAttention builds a KV-cache management inspired from the paging technique used in operating systems. CachedAttention [\[13\]](#page-13-24) implements a hierarchical KV caching system for multi-turn conversations of LLMs. InfiniGen [\[23\]](#page-13-25) implements KV-cache management for long-text generation. These systems are

designed to optimize KV-cache storage by minimizing waste and enabling flexible sharing of KV-cache resources both within and across requests, thereby substantially reducing memory utilization.

System-aware model optimization. FlashAttention series [\[10,](#page-13-26) [11,](#page-13-27) [29\]](#page-13-28) speed up the multi-head attention calculation with io-aware operator execution. FasterTransformer [\[1\]](#page-12-5) improves the GPU utilization for transformers with modelspecific computation-aware GPU kernel implementations. Shibo et al. [\[36\]](#page-13-29) propose a communication-aware decomposition technique to overlap communication with dependent computation.

LLM serving scheduling. FastServe [\[38\]](#page-14-6) introduces a novel preemptive scheduling technique that allows for token-level preemption, effectively reducing delays caused by head-ofline blocking. DistServe [\[45\]](#page-14-7) enhances LLM serving by separating the prefill and decoding stages into different computational processes. Llumnix [\[32\]](#page-13-30) aims to react to heterogeneous and unpredictable requests by online rescheduling across multiple model instances. VTC [\[30\]](#page-13-31) gives a definition of LLM serving fairness and proposes a Virtual Token Counter scheduling policy to ensure the fair processing of all clients.

