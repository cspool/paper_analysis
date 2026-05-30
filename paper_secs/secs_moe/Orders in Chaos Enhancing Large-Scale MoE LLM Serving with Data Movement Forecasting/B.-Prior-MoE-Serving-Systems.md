# B. Prior MoE Serving Systems

The MoE mechanism constitutes the primary source of data movement overhead in modern serving systems. As illustrated in Figure 2, take DeepSeek V3 as an example, MoE-related data movement (MoE All-to-All and MoE Weights) dominates the overhead across different serving configurations, accounting for 60%-90% of total latency under 4K sequence length. To address this, existing research has developed numerous system-level solutions targeting different performance and cost objectives. Edge systems like MoE-Lightning [31] and CoServe [32] employ CPU memory offloading techniques to address GPU memory capacity constraints, while cloud systems such as Comet [9] and MegaScale-Infer [15] target multi-GPU systems and address GPU-GPU communications in MoE for higher throughput. Novel hardware architectures like Duplex [33] explore processing-in-memory to accelerate data movement in MoE LLMs.

However, these prior studies employ a *system-centric* methodology when optimizing for MoE LLMs. Namely, they inherently focus on a specific platform and the corresponding data movement patterns of MoE in such platform (e.g., CPU-GPU, multi-GPU, ML accelerators). As a result, they propose deployment-specific optimizations that may not generalize across different serving platforms, and their insights are often a slice of the overall inherent patterns in MoE LLMs.

In this work, we flip the process and adopt a *model-centric* strategy by conducting system-independent profiling to extract *system-agnostic* insights about MoE data movement patterns. These insights are therefore broadly applicable across various platforms, providing a foundation for optimization strategies that transcend specific system implementations.

#### III. MOE PROFILING AND SYSTEM INSIGHTS

In this section, we conduct a data-movement-centric profiling of the expert selection behavior in four state-of-the-art MoE models: Deepseek V3 (671B), Llama4-Maverick-128E (402B), Qwen3-235B (235B), and Kimi K2 (1000B). All results are averaged over more than 24,000 requests.

