# <span id="page-12-9"></span>**A GPU Memory Usage During Inference**

During LLM inference, the prefill stage—where the entire input sequence is processed at once—dominates GPU memory usage due to the storage of intermediate activations and key-value (KV) cache across all tokens. In contrast, the decode stage generates output token by token, reusing the KV cache from previous steps, which results in significantly lower memory consumption as the model only processes one token at a time, as indicated in Figure [9.](#page-13-0)

![](_page_13_Figure_1.jpeg)

<span id="page-13-0"></span>Figure 9: GPU Memory Usage During Inference: starting from the second datapoint, each datapoint represents the memory usage when generating a new token. The memory peaks before generating the first token and drops significantly during decode stage.

## <span id="page-13-2"></span>B Basic Chunked Prefill Algorithm

Chunked prefill is an alternative technique for reducing inference memory by splitting the context into smaller chunks during the prefill stage. While more complex implementations can also improve computational speed, we compare it with the simplest version (See Algorithm 2), which is primarily designed to reduce memory usage.

```
Algorithm 2 Basic Chunked Prefill
```

```
Require: Input sequence X \in \mathbb{R}^{B \times S \times d}, chunk size C, large language model M Initialize empty key-value cache K Split X into chunks: X^{(1)}, X^{(2)}, \ldots, X^{(\lceil S/C \rceil)} where each X^{(i)} \in \mathbb{R}^{B \times C \times d} has at most C tokens for each chunk X^{(i)} do Compute model Output^{(i)} = M(X^{(i)}, K) Extract and store key-value pairs in cache: K \leftarrow K \cup KV(Output^{(i)}) end for Proceed with normal autoregressive decoding using cached K
```

