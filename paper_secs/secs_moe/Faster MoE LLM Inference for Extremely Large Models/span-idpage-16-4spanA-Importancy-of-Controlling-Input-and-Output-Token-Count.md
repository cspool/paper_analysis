# <span id="page-16-4"></span>**A Importancy of Controlling Input and Output Token Count**

We conducted tests under a concurrent request setting of 512, ensuring that the total number of input and output tokens remained fixed at 2048. The results of our evaluation are presented in Table [3](#page-16-5) for Deepseek-V2-Lite. Some datapoints on Deepseek V3 is presented in Table [4.](#page-16-6)

From this experiment, we observe that increasing the proportion of input tokens can significantly enhance overall throughput. This is primarily due to the fact that the prefill phase is highly parallelizable, with the computational cost evenly distributed across all tokens. In contrast, autoregressive decoding operates sequentially, processing only one token at a time, which leads to lower efficiency.

For the same reason, all our efficiency evaluations are conducted under this fixed setting rather than relying on performance benchmarks based on real-world workload scenarios. The latter approach introduces uncontrollable variables and lacks accuracy in assessing computational efficiency.

<span id="page-16-5"></span>

| Input token  | 256  | 512  | 768  | 1024 | 1280  | 1536  | 1792  |
|--------------|------|------|------|------|-------|-------|-------|
| Output token | 1792 | 1536 | 1280 | 1024 | 768   | 512   | 256   |
| Throughput   | 6368 | 7106 | 8224 | 9484 | 10419 | 11584 | 13040 |

<span id="page-16-6"></span>Table 3: Influence of IO token counts on throughput for Deepseek-V2-Lite.

| Input token  | 1024 | 1024 |
|--------------|------|------|
| Output token | 1024 | 8    |
| Throughput   | 2636 | 8487 |

Table 4: Influence of IO token counts on throughput for Deepseek-V3.

