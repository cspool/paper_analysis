# Abstract

Diffusion Models (DMs) have demonstrated remarkable performance in a variety of image generation tasks. However, their complex architectures and intensive computations result in significant overhead and latency, posing challenges for hardware deployment. To address these issues, researchers have explored the sparsity in DMs to reduce computational workloads, including semantic sparsity in image generation and spatial sparsity in local editing. Unfortunately, existing sparsity prediction methods face critical limitations in deployment: 1) additional prediction overheads offset the benefits of sparsity; 2) convolution and general matrix multiplication (GEMM) exhibit distinct sparsity patterns, which current co-design frameworks struggle to process. In this paper, we introduce S-DMA, a software-hardware co-design framework that unifies efficient sparsity prediction while supporting various sparse operators. First, we propose a spatiality-aware similarity computation method that leverages the local similarity of images, reducing the computational complexity of sparsity prediction from O( 2 ) to O(N). Second, we implement NAND-based similarity for sparsity prediction, which minimizes the computational overheads and ensures adaptability to different sparsity schemes. Finally, a dedicated hardware architecture is designed to efficiently leverage the algorithm optimizations. A NAND-based sparsity prediction processing unit is designed to adaptively handle the sparsity patterns. Additionally, a sparsity-aware reduction network and a dimension-adaptive

Bo Liu is the corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MICRO '25, Seoul, Republic of Korea

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1573-0/25/10 <https://doi.org/10.1145/3725843.3756046>

dataflow are employed to support convolution and GEMM with different DM sparsity patterns. Experimental results demonstrate that S-DMA achieves up to 51.11× speedup and 43.87× higher energy efficiency than NVIDIA A100 GPU. Compared to state-of-the-art DM accelerators, S-DMA achieves up to 7.05× speedup and 3.19× higher energy efficiency.

