# Abstract

While Large Language Models (LLMs) are widely adopted, their massive parameter size constrains practical deployment. A common solution is clustering-based non-uniform quantization, which effectively compresses models to as low as 3 bits per weight while preserving high accuracy. However, instead of accelerating memory-bound LLM inference, the memory reduction paradoxically often causes a significant slowdown due to dequantization overhead and GPU underutilization. To address the issue, we propose Quantix, a framework designed to convert memory savings into inference speedups. Quantix applies two key optimizations: (1) a hardware-aligned bit shuffling scheme for efficient data access, and (2) a fused dequantization-multiplication pipeline that effectively maps workloads on both CUDA and Tensor Cores. Quantix enables high-throughput batched inference, delivering average kernel-level speedups of 4.82× over FP16 cuBLAS and end-to-end speedups of up to 11.46× over stateof-the-art quantization methods on NVIDIA L40 GPUs.

CCS Concepts: • Computing methodologies → Parallel computing methodologies; Natural language processing.

Keywords: Large language model (LLM) inference, GPU programming

#### ACM Reference Format:

YuAng Chen, Wenqi Zeng, and Jeffrey Xu Yu. 2026. High-Throughput Non-uniformly Quantized 3-bit LLM Inference. In Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '26), January 31 – February 4, 2026, Sydney, NSW, Australia. ACM, New York, NY, USA, [13](#page-12-0) pages. <https://doi.org/10.1145/3774934.3786423>

