# <span id="page-18-1"></span>D.5. Results on Warm-up

To verify that warm-up is not the primary source of improvement, we conduct an ablation study by applying warm-up to the baseline and removing it from MoDiff. The experiments are performed using the DDIM sampler on CIFAR-10 with LCQ. As shown in Table [18,](#page-18-3) MoDiff consistently outperforms the baseline under fair comparison, indicating that the observed performance gains are not attributable to the warm-up mechanism.

Table 18. FID on CIFAR-10 using the DDIM sampler in the ablation study of warm-up. The best performance is bolded.

<span id="page-18-3"></span>

| Bits (W/A) | LCQ w/o warmup | LCQ w/ warmup | LCQ+MoDiff w/o warmup | LCQ+MoDiff w/ warmup |
|------------|----------------|---------------|-----------------------|----------------------|
| 8/8        | 4.19           | 4.19          | 4.22                  | 4.21                 |
| 8/6        | 9.93           | 9.53          | 4.25                  | 4.00                 |
| 8/4        | 306.06         | 299.96        | 31.22                 | 28.19                |

Moreover, as indicated by Theorem [4.4,](#page-5-3) warm-up can be achieved by repeatedly inputting a<sup>T</sup> . This process converges to the full-precision activation due to the contraction of the quantization error. As demonstrated in our experiments, approximately 4 to 5 steps are sufficient to reduce the quantization error to a negligible level on CIFAR-10 using 4-bit precision.

## <span id="page-18-0"></span>D.6. Analysis on Memory Consumption

In the main paper, we present the trade-off analysis between computation cost and memory cost for MoDiff when generating a single image on CIFAR-10 with DDIM. In this section, we extend our analysis to larger batch sizes selected from {2, 4, 8}. The results are shown in Tables [19,](#page-18-4) [20,](#page-18-5) and [21.](#page-19-2) The results, shown in Tables [19,](#page-18-4) [20,](#page-18-5) and [21,](#page-19-2) demonstrate that MoDiff significantly reduces computation cost while incurring only a minimal increase in memory usage.

<span id="page-18-4"></span>Table 19. The relationship between BOPs and memory usage of our method using DDIM on CIFAR-10 for generation with batch size 2.

| Measurement | W8A2  | W8A4  | W8A8  | W8A32 |
|-------------|-------|-------|-------|-------|
| GBops       | 204   | 410   | 918   | 3272  |
| Memory (Mb) | 36.49 | 38.89 | 43.69 | 36.09 |

<span id="page-18-5"></span>Table 20. The relationship between BOPs and memory usage of our method using DDIM on CIFAR-10 for generation with batch size 4.

| Measurement | W8A2  | W8A4  | W8A8  | W8A32 |
|-------------|-------|-------|-------|-------|
| GBops       | 408   | 820   | 1836  | 6544  |
| Memory (Mb) | 38.89 | 43.69 | 53.28 | 38.09 |

<span id="page-19-2"></span>Table 21. The relationship between BOPs and memory usage of our method using DDIM on CIFAR-10 for generation with batch size 8.

| Measurement | W8A2  | W8A4  | W8A8  | W8A32 |
|-------------|-------|-------|-------|-------|
| GBops       | 906   | 1640  | 3672  | 13088 |
| Memory (Mb) | 43.69 | 53.28 | 72.47 | 42.09 |

