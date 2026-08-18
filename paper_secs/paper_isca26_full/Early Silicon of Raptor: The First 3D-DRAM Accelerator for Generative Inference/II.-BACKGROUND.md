# II. BACKGROUND

*A. Transformer Models: An Overview of Generative Inference*

We focus on decoder-only Transformers, the dominant backbone for generative AI [84]. They generate sequences autoregressively, factorizing the joint probability as

$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{< t}),$$

where x1:<sup>T</sup> = (x1, . . . , x<sup>T</sup> ) is a sequence of T tokens from vocabulary V and x<t denotes the prefix up to position t − 1. Each token depends on all prior tokens, so decoding is inherently sequential. This dependency leads to two distinct phases with different system behavior. The prefill phase processes the input context with large, highly parallel matrix multiplications and exhibits strong weight reuse. The decode phase produces one token at a time and repeatedly reads and writes the key–value (KV) cache, driving high memory traffic with limited reuse. Prefill is compute-bound; decode is dominated by memory bandwidth and capacity. These access patterns set the performance envelope for generative serving.

