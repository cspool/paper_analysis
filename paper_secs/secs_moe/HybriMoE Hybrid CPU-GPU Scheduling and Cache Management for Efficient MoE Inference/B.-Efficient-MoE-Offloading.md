# B. Efficient MoE Offloading

Parameter-offloading techniques have been proposed to address the significant memory requirements of large language models (LLMs) [20], [21]. However, these techniques are primarily designed for dense models and involve loading or prefetching all parameters, leading to unnecessary communication overhead. To accommodate the sparse

TABLE I
COMPARISON WITH EXISTING OFFLOADING WORKS.

<span id="page-1-1"></span>

|            | Offload     | CPU            | Dynamic | Cache        |
|------------|-------------|----------------|---------|--------------|
|            | Granularity | Computation    | Mapping | Optimization |
| Powerinfer | Neuron      | Decode         | ✓       | LFU          |
| llama.cpp  | Layer       | Prefill+Decode | ×       | LFU          |
| AdapMoE    | Expert      | ×              | ✓       | LRU          |
| KTrans     | Expert      | Decode         | ×       | LFU          |
| Ours       | Expert      | Prefill+Decode | ✓       | Score-Aware  |

activation patterns in Mixture-of-Experts (MoE) models, several specialized techniques have been introduced, including advanced gating, prefetching, and quantization strategies [3]–[7], [22]–[26]. These methods aim to minimize the on-demand loading overhead, reducing unnecessary memory transfers and improving overall performance.

## C. Hybrid CPU-GPU Scheduling

Previous offloading techniques have primarily focused on reducing memory transfer overhead by offloading certain computations to the CPU [9]. For instance, PowerInfer [10] reduces GPU memory demand by executing less frequently activated neurons on the CPU, taking advantage of skewed activation patterns. Caraserve [11] addresses cold-start delays in LoRA serving by utilizing CPU assistance and employing rank-aware scheduling to reduce latency. These methods are effective in scenarios where activations are skewed or tasks have long periods of parameter reuse.

In the context of MoE models, techniques like Fiddler [12] and kTransformers [13] extend this concept by offloading expert layer computation to the CPU during cache misses. Specifically, when an expert is not in the GPU cache, the CPU executes the corresponding expert layer instead of loading it from memory. These approaches aim to optimize memory usage by exploiting CPU-GPU parallelism and mitigating the overhead of loading large models onto the GPU.

In table I, we compare HybriMoE with prior-art works qualitatively. As can be observed, HybriMoE features CPU-GPU hybrid scheduling to improve the efficiency of both prefill and decode stages.

