# D. Real-world Application Study: DeepSeek-R1 Inference System

In this section, we demonstrate that STAGE can model real-world LLM workloads using the DeepSeek-R1 inference architecture [13], which separates prefilling and decoding. These two phases exhibit distinct performance characteristics and require different parallelism configurations.

We evaluate a system with 144 GPUs, partitioned into either 4 clusters of 36 GPUs, 2 clusters of 72 GPUs, or a single 144-GPU cluster. Within each cluster, we use expert parallelism for MoE layers and data parallelism for the remaining layers. The total batch size across clusters is fixed at 2048. The resulting decoding and prefilling performance under different EP degrees is shown in Table VIII.

Prefilling generally prefers lower EP degrees because it operates on long sequences and large batches, making it compute-bound while reducing all-to-all overhead. Conversely, decoding handles short sequences per step and benefits from larger effective batch sizes, thus achieving higher throughput with larger clusters and higher EP degrees.

