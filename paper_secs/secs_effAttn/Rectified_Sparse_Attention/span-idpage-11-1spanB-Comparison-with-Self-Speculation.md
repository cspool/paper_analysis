# <span id="page-11-1"></span>B Comparison with Self-Speculation

As discussed in Section [4,](#page-8-1) ReSA shares similar computational characteristics with sparse KV cachebased self-speculation. The rectification phase in ReSA resembles the verification phase used in self-speculative methods. However, unlike these methods, ReSA does not rely on output logits to make per-token accept / reject decisions. This design choice is motivated by the observation that, when sparse attention achieves high generation quality, this kind of token-wise strict verification can significantly increase latency without providing proportionate accuracy gains.

To validate this, we compare ReSA and sparse KV-based self-speculation on mathematical reasoning tasks. We set the speculation length to 16, meaning that the model drafts 16 tokens using the sparse KV cache. Similarly, we set ReSA's rectification frequency to 16. Across all tasks, ReSA achieves nearly 2× speedup over self-speculation while maintaining comparable accuracy. This is because, in each verification step of speculative decoding, only about 8 tokens are typically accepted—effectively halving the generation rate compared to ReSA. Although this strict verification ensures that speculative decoding matches the accuracy of dense attention, we have previously shown that ReSA also approaches the accuracy of dense attention. Therefore, we believe that the marginal accuracy gains of speculative decoding do not justify its substantial latency overhead.

| Task          | Sparse KV Self-Spec. | Rectified Sparse Attention |
|---------------|----------------------|----------------------------|
| Minerva       | 1×                   | 1.93×                      |
| Gaokao2023En  | 1×                   | 1.87×                      |
| OlympiadBench | 1×                   | 1.98×                      |
| AIME24        | 1×                   | 1.96×                      |
| AMC23         | 1×                   | 1.86×                      |
| Average       | 1×                   | 1.92×                      |

Table 3: Decoding speedup comparison. We set the throughput of self-speculation as baseline. ReSA achieves larger speedup compared with sparse self-speculative decoding.