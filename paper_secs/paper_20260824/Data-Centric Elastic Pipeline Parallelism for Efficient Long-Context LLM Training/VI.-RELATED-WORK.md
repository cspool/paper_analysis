# VI. RELATED WORK

*Long Context Training:* Many sequence parallelism patterns for long context training have been proposed [\[7\]](#page-11-5), [\[21\]](#page-11-7), [\[25\]](#page-11-8), which can be used to replace Ulysses-style SP and are orthogonal to our method. Other works [\[15\]](#page-11-27), [\[35\]](#page-12-4) observe the skewness distribution of sequence length and aim to address workload heterogeneity. These works are orthogonal because optimization for PP and checkpointing are not considered.

*Pipeline Parallelism Optimization:* Recent works like AdaPipe [\[32\]](#page-12-5), Mario [\[26\]](#page-11-28) have explored checkpointing and offloading optimizations with PP. However, these works assume homogeneous workloads, but we focus on heterogeneous workloads with varied-length input. ByteScale [\[14\]](#page-11-13) and WLB-LLM [\[36\]](#page-12-3) optimize workload balance of batch-level PP for heterogeneous workload, ignoring the optimization opportunity of sequence splitting, restricting its applicability in longcontext training scenario.

