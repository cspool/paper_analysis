# E.4 EFFICIENCY ACROSS VARIOUS HARDWARE

<span id="page-21-2"></span>Table 13: Inference latency (millisecond) of LLaVA-Mini on various hardware platforms.

| Methods    | #Vision Tokens | RTX 3090 (24G) | A100 (40G) | A800 (80G) |
|------------|----------------|----------------|------------|------------|
| LLaVA-v1.5 | 576            | 198.75         | 113.04     | 87.43      |
|            | 1              | 64.52          | 38.64      | 27.43      |
|            | 4              | 65.52          | 38.84      | 27.71      |
| LLaVA-Mini | 16             | 68.97          | 39.28      | 28.92      |
|            | 64             | 80.10          | 46.23      | 34.65      |

The efficiency improvements brought by LLaVA-Mini stem from reduced computational load (FLOPs), which is consistent across different hardware platforms. To demonstrate the scalability of model efficiency across different hardware platforms, we compute the inference latency of LLaVA-Mini on three hardware platforms: RTX 3090, A100, and A800. As shown in Table [13,](#page-21-2) the efficiency improvements brought by LLaVA-Mini are scalable across these hardware platforms.

