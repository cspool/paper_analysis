# **A** Discussion of Generation Time Speedup

To better assess SnapKV's effectiveness across different stages, we documented a detailed time breakdown for Mistral-7B-Instruct-v0.2 during both the prompting and generation stages. We configured the model to consistently generate 512 tokens, facilitating a direct comparison with the prompting stage. As illustrated in Figure 10, generation time dominates the whole processing time for LLMs over input sequences, introducing significant overhead. While the generation time for the original model increases with input length, SnapKV maintains a consistent decoding speed regardless of input length, significantly reducing generation time. Especially, SnapKV is able to achieve balanced prompting time and generation time with input length smaller than 100k.

<span id="page-15-0"></span>![](_page_15_Figure_2.jpeg)

Figure 10: The prompting time and generation time comparison between Mistral model with and without SnapKV.

