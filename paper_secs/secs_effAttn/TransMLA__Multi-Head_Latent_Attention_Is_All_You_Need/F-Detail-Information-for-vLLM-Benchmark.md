# F Detail Information for vLLM Benchmark

In Section [5.4,](#page-10-1) we demonstrated the speedup achieved by TransMLA—which compresses 92.97% of the KV cache—compared to the original LLaMA-2-7B model. This section provides a detailed analysis of throughput across various hardware configurations.

To account for the effects of both the prefilling and decoding stages, we adopt a setting where the input and output lengths are equal. For instance, with a total context length of 1k, we set the input length to 512 tokens and the output length to 512 tokens. Most experiments are conducted using 100 requests to compute the average throughput. However, for shorter context lengths such as 1k, inference is extremely fast, leading to some timing fluctuations. To mitigate this, we increase the number of requests to 1000 for more stable measurements.

While the original LLaMA-2-7B model supports a maximum context length of 4096 tokens, we extend this limit to 32k tokens in our evaluation. Detailed throughput results are presented in Table [4.](#page-25-0)

On a GPU with 165.2 TFLOPS of compute and 24GB of memory, the LLaMA-2-7B model runs out of memory when the context length reaches 16k tokens. In contrast, TransMLA sustains a throughput of 414.41 tokens per second under the same conditions. On a more powerful GPU with 320 TFLOPS and 64GB of memory, we employ a development version of the vLLM framework. We anticipate that the throughput of TransMLA will improve further with the release of future optimized versions of the framework tailored for this hardware.

<span id="page-25-0"></span>Table 4: Throughput comparison between LLaMA-2-7b and TransMLA at varying input lengths and number of requests.

|                | Requests |            | Throughput(output tokens/s) |             |             |  |
|----------------|----------|------------|-----------------------------|-------------|-------------|--|
| Context Length |          | Model      | 165.2 TF 24GB               | 312 TF 40GB | 320 TF 64GB |  |
| 1K             | 1000     | LLaMA-2-7b | 653.81                      | 1579.26     | 1249.13     |  |
|                |          | TransMLA   | 3043.65                     | 4062.43     | 1798.17     |  |
| 2K             | 100      | LLaMA-2-7b | 352.85                      | 850.14      | 789.31      |  |
|                |          | TransMLA   | 2241.87                     | 2577.01     | 1080.73     |  |
| 4K             | 100      | LLaMA-2-7b | 173.09                      | 441.37      | 442.63      |  |
|                |          | TransMLA   | 1318.78                     | 1926.15     | 1021.03     |  |
|                | 100      | LLaMA-2-7b | 85.80                       | 218.51      | 216.66      |  |
| 8K             |          | TransMLA   | 832.69                      | 1118.18     | 870.15      |  |
| 16K            | 100      | LLaMA-2-7b | OOM                         | 110.58      | 112.13      |  |
|                |          | TransMLA   | 414.41                      | 601.36      | 483.22      |  |
|                | 100      | LLaMA-2-7b | OOM                         | 38.32       | 55.69       |  |
| 32K            |          | TransMLA   | OOM                         | 243.81      | 278.09      |  |

