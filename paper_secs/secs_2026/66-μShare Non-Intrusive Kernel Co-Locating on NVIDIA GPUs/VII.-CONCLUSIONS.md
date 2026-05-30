# VII. CONCLUSIONS

In this paper, we analyze the semantic gap between the resource demands of kernels and the resource allocation performed by the NVIDIA GPU hardware scheduler. This gap leads to stacked co-location of kernels and results in low utilization of low-level hardware resources. Without intrusive modifications to GPU, we propose a hardware–software codesign approach, *half-plus blocksize shaping*, which achieves scattered co-location of kernels. Building on this concept, we construct the *μShare* system, which effectively improves GPU resource efficiency across various NVIDIA GPUs and diverse co-location scenarios.

