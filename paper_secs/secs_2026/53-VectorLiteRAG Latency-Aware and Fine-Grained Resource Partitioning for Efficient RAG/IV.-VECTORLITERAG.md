# IV. VECTORLITERAG

VECTORLITERAG is an optimized RAG system that determines the optimal configuration for a CPU–GPU hybrid vector index. It is organized around tightly integrated components: (1) performance modeling and latency-aware hybrid index construction, and (2) a distributed runtime pipeline for inference serving. Given the latency constraint, LLM, index, and system configuration, VECTORLITERAG computes a partitioning point for tiered search, constructs the hybrid index, and serves inference requests through a tailored pipeline.

Hybrid Index Construction. The first component of VEC-TORLITERAG focuses on understanding the performance characteristics of the underlying system. This stage profiles CPU-based search latency, query-to-cluster access patterns, and standalone LLM throughput to characterize contention between retrieval and generation. These measurements drive a performance model and cache-coverage estimator, enabling a latency-bounded partitioning algorithm to select hot clusters. The hot clusters are then sharded into GPU sub-indexes.

Distributed VECTORLITERAG Pipeline. The second component is the runtime pipeline that operationalizes the hybrid index. At runtime, batched queries are routed to CPU or GPU shards using mapping tables generated during index construction, allowing each shard to operate with a flexible nprobe budget and reducing contention with LLM. A dynamic dispatcher further improves batching efficiency by advancing early-completing queries to mitigate tail latency.

The partitioning scheme and runtime pipeline are independent of the distance metric or compression method. As long as the index exhibits clustered structure and benefits from GPU acceleration, VECTORLITERAG can identify an effective hybrid configuration and deliver SLO-compliant RAG service.

