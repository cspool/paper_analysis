# **7 Discussion**

We create BenchPress, an evaluation suite that provides a controlled, reproducible framework, not tied to any single compression paradigm, for measuring progress in this space. This is largely motivated by inconsistent evaluation practices in existing work, which entail comparison challenges, for example in not accounting for the capabilities of simple baselines, as our experiments show.

We show that the causal compression-token approach is a weak baseline: mean pooling and bidirectional compression tokens both leverage bidirectional encoding to achieve considerably stronger results. This convergence suggests that causal attention is a poor inductive bias for compression, pointing toward encoder-style or prefix-LM backbones as more natural starting points. Interestingly, bidirectional tokens benefit from multi-ratio training while mean pooling generally does not, plausibly because forward attention gives the model an explicit view of its compression budget. Testing this hypothesis directly is a promising direction for future work.

A persistent gap between all methods and the teacher at 128× suggests that advances in compressor architecture are needed alongside scaling. We hope these standardized practices and strong baselines provide a foundation for future work across compression paradigms.

