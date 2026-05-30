# Size Zheng

Bytedance Seed Beijing, China zheng.size@bytedance.com

#### Haibin Lin

Bytedance Seed Bellevue, WA, United States haibin.lin.cmu@gmail.com

## Xin Liu

Bytedance Seed Bellevue, WA, United States liuxin.ai@bytedance.com

#### **Abstract**

Low-latency, single-request decoding of large language models is critical for interactive systems with tight SLA demands. Prior work reduces latency through speculative decoding (combining a small *draft* model with a larger *target* model), but the draft model remains on the critical path, and communication overhead limits scaling across GPUs due to the small batch size associated with single-request decoding. To address these limitations, this paper introduces SwiftSpec: a system architecture that disaggregates draft and target models across homogeneous GPUs within a single node and utilizes NCCL-low-latency primitives directly to improve the performance of core GEMM and attention kernels. Our implementation includes 3k lines of custom CUDA for fused kernels and an evolving tree cache for KV-cache consistency and maximized reuse between draft and target models. On a single 8×H800 GPU node, SwiftSpec achieves 347 tokens/s for Llama-3-70B-1.3× faster than NVIDIA's own benchmarks on a higher-performance 8×H200 setup-and averages 1.75× faster decoding than state-of-the-art speculative decoding across five model families and six datasets. Specifically, we find that for Llama-3-70B SwiftSpec is significantly faster across all 480 tested queries, showing 1.7×

\*Work partially done at University of Chicago

![](_page_0_Picture_18.jpeg)

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

ASPLOS '26, Pittsburgh, PA, USA.

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 https://doi.org/10.1145/3779212.3790246

