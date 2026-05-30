# <span id="page-26-0"></span>D.6 INITIALIZATION OVERHEAD OF ADAPTERS

Initialization Time. Table [17](#page-26-1) reports the initialization latency of the low-rank adapter (CLoQ) and the FT-based adapters (QWHA and SSH) under the 4-bit setting across different models. The initialization of CLoQ requires gathering activations and quantization errors, followed by SVD decomposition before solving the least-squares problem. In contrast, the initialization of FT-based adapters involves collecting activations and quantization errors, followed by parameter selection and value assignment through channel-wise transforms and solving least-squares problem. During this process, the fast Hadamard kernel allows QWHA with WHT to perform efficient computation. As a result, QWHA achieves comparable initialization time to CLoQ, while SSH with the DHT kernel incurs significantly higher latency.

<span id="page-26-1"></span>Table 17: Initialization latency (hours) of each method under the 4-bit setting.

| Method | LLaMA-3.2-3B | LLaMA-3.1-8B | Mistral-7B-v0.3 |
|--------|--------------|--------------|-----------------|
| CLoQ   | 0.58         | 1.14         | 1.26            |
| SSH    | 3.85         | 8.09         | 8.58            |
| QWHA   | 0.66         | 1.34         | 1.46            |

Memory Usage. We measure the peak memory consumption during initialization for each method on the LLaMA-3.2-3B model with 4-bit quantization, broken down into cached Hessians and model weights (Table [18\)](#page-26-2). While every methods require multiple matrix projections, CLoQ additionally performs SVD, which leads to higher memory usage in layers with large dimensions. Overall, QWHA achieves slightly lower memory footprint due to its efficient block-wise computational implementation of the fast Hadamard transform.

<span id="page-26-2"></span>Table 18: Memory usage (GB) and component breakdown during initialization of each method.

| Method | Total Usage | Model Weight | Cached Hessian |
|--------|-------------|--------------|----------------|
| CLoQ   | 12.77       | 3.04         | 7.10           |
| SSH    | 11.93       | 3.04         | 7.10           |
| QWHA   | 11.52       | 3.04         | 7.10           |

