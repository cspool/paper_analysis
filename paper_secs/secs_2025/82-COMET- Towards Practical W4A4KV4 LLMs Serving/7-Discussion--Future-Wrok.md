# 7 Discussion & Future Wrok

COMET is a pioneering mixed-precision LLM inference framework that utilizes data distribution characteristics to achieve optimized end-to-end performance. Its current implementation focuses on optimizing the GEMM kernel, yielding significant speedups in throughput. Our evaluation shows that the proposed COMET-W4Ax kernel performs near the theoretical upper bound, demonstrating the effectiveness of COMET's design for mixed-precision computations.

Moving forward, we aim to further enhance COMET by incorporating attention kernel optimizations. In LLM inference, GEMM and attention computations occupy approximately 65% and 32% of the total runtime, respectively [33]. While COMET has substantially optimized GEMM performance, attention computation remains a critical component of the runtime. Previous studies have introduced efficient methods for optimizing attention during both prefill and decode phases [9, 17, 52], using algorithmic transformations to

reduce data transfer costs. These attention optimizations operate independently of COMET-W4Ax and offer a promising next step for further performance gains.

In addition to attention kernel improvements, we plan to integrate COMET with various compilation and scheduling optimizations developed in LLM serving systems. Techniques such as operator-level pipelining and intra-device parallelism [3, 16, 23, 43, 57, 66, 67] complement COMET and could enhance end-to-end performance. Integrating these scheduling strategies with COMET's framework may unlock further efficiencies, making it a robust solution for high-performance LLM inference.

#### 8 Conclusion

In this paper, we present COMET, the first mixed-precision LLM inference framework designed for practical W4A4KV4 LLM serving, primarily built on the proposed FMPQ algorithm and the COMET-W4Ax kernel. Specifically, the FMPQ algorithm efficiently achieves low-precision quantization for activations and the KV cache with minimal accuracy loss. Moreover, the open-source COMET-W4Ax kernel can be seamlessly integrated into existing inference systems. It includes optimizations for data layout, GPU software pipeline, and streaming multiprocessor scheduling, addressing data access and load imbalance issues in mixed-precision GEMM on modern GPUs. Evaluations on a single A100-80G-SXM4 demonstrate that COMET achieves a 2.02× end-to-end throughput improvement over state-of-the-art baselines, showcasing its potential for enhancing LLM inference efficiency.

