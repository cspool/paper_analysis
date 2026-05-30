# <span id="page-9-6"></span>4 CONCLUSION AND FUTURE WORK

In this paper, we introduced LogQuant, a novel quantization technique designed to optimize KV Cache management in large language models (LLMs). Our approach leverages a base-2 logarithmic strategy to maintain sparsity while accommodating an increased number of full-precision tokens. Through comprehensive evaluations, we demonstrated that LogQuant consistently outperforms existing methods, such as KiVi, across various model families and compression ratios, particularly benefiting smaller models that typically suffer from accuracy loss due to quantization.

We further explored the efficiency of our implementation within the HuggingFace pipeline, achieving notable improvements in throughput and memory utilization. Additionally, our investigation into accuracy loss across different tasks highlighted LogQuant's superior retention of performance, especially in complex tasks. These findings underscore the potential of LogQuant to enhance LLM inference in resource-constrained environments.

Future work will focus on refining our quantization approach and investigating further optimizations, such as operator fusion, to maximize efficiency and performance in LLM applications.

