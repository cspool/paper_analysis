# *A. Decode is the industry-wide bottleneck for LLM inference*

Decoder-only Transformers generate tokens autoregressively and therefore execute a highly-parallel *prefill* followed by a sequential *decode* stage (formalized in §II-A). In production LLM serving, decode commonly dictates both *tail latency* and *fleet throughput* because each output token triggers attention over an ever-growing history and repeatedly accesses the key–value (KV) cache. This is not merely an academic observation: major inference stacks now explicitly target *KVcache bottlenecks* via paging (e.g., PagedAttention), KV reuse, and KV quantization, indicating that KV bandwidth/capacity is the limiting resource in practice [30], [35], [58], [59].

