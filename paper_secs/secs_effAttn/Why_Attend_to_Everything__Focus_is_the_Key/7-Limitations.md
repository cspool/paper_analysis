# 7 Limitations

The limitations of our Focus are as follows.

Training cost. Soft gating computes all O(n 2 ) pairs during training, so efficiency is inference-only for now. Training directly with discrete assignments remains open.

Quality benefit diminishes with scale. Focus surpasses full attention at 124M but only matches it at 774M–1.5B (within 0.3–0.4 PPL). Although this is good for a sparse model, it seems larger models are less susceptible to noisy attention patterns. The good thing is that the efficiency benefit (speedup) still grows with sequence length regardless of scale.

Routing overhead at short sequences. Sorting and gather/scatter add ∼12ms constant overhead, which dominates at sequences ≤4K. Focus offers no speedup below 16K tokens.

