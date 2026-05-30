# <span id="page-20-1"></span>**L PyramidKV will cause minimal extra inference overhead.**

The allocation strategy and score-based selection add minimal complexity in the inference phase compared to the computation required for next-token predictions as [Table 9.](#page-20-2) Each row shows the setting of using a specific "[Prompt length, Generation length]" combination. We show the inference speed comparison between total inference time, time for allocation strategy and time for score-based selection on LlaMa-3-8B-Instruct. Each cell is the latency measured in seconds. Furthermore, our budget allocation can be calculated before inference, requiring only a one-time computation. Thus, PyramidKV will cause minimal extra inference overhead.

| Prompt Length | Generation Length | Inference Time | Allocation Time | Selection Time |
|---------------|-------------------|----------------|-----------------|----------------|
| 512           | 512               | 18.26          | 0.0000003       | 0.0194         |
| 512           | 1024              | 34.69          | 0.000002        | 0.0133         |
| 512           | 2048              | 70.69          | 0.000003        | 0.013          |
| 512           | 4096              | 138.62         | 0.000005        | 0.013          |
| 1024          | 512               | 17.32          | 0.000002        | 0.0131         |
| 1024          | 1024              | 34.67          | 0.000002        | 0.01288        |
| 1024          | 2048              | 70.21          | 0.000005        | 0.01296        |
| 1024          | 4096              | 138.61         | 0.000003        | 0.01297        |
| 2048          | 512               | 17.48          | 0.000004        | 0.0128         |
| 2048          | 1024              | 34.78          | 0.000006        | 0.0129         |
| 2048          | 2048              | 69.50          | 0.000003        | 0.01297        |
| 2048          | 4096              | 138.59         | 0.000003        | 0.013          |
| 4096          | 512               | 17.58          | 0.000002        | 0.013          |
| 4096          | 1024              | 34.93          | 0.000004        | 0.0129         |
| 4096          | 2048              | 69.65          | 0.000002        | 0.013          |
| 4096          | 4096              | 138.87         | 0.000002        | 0.013          |

<span id="page-20-2"></span>Table 9: Extra inference overhead of PyramidKV

