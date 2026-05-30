# 6 Related Work

Efficient attention methods fall into three families. Sparse attention methods (Longformer [\[Beltagy](#page-11-0) [et al.,](#page-11-0) [2020\]](#page-11-0), BigBird [\[Zaheer et al.,](#page-13-0) [2020\]](#page-13-0)) use fixed positional patterns with exact softmax. They cannot adapt to content and degrade quality when retrofitted. Linear attention (Performer [\[Choromanski et al.,](#page-11-1) [2021\]](#page-11-1)) replaces softmax with kernel approximations; it diverges catastrophically in the retrofit setting (+75.6 PPL). Low-rank attention (Linformer [\[Wang et al.,](#page-13-1) [2020\]](#page-13-1)) projects keys/values to fewer positions but is incompatible with causal modeling.

Routing Transformer [\[Roy et al.,](#page-12-1) [2021\]](#page-12-1) is our closest prior work—both use content-based routing. Key differences: (1) online k-means (transient) vs learned centroids (stable); (2) replaces attention mask vs gates existing attention; (3) no balancing vs Sinkhorn.

Mixture of Experts [\[Fedus et al.,](#page-11-3) [2022\]](#page-11-3) and Focus both route computation via learnable parameters, but MoE routes tokens to FFN experts while Focus routes attention connections. The two are complementary; our Sinkhorn solves the analogous load-balancing problem.

Token selection methods [\[Ribar et al.,](#page-13-2) [2024,](#page-13-2) [Chen et al.,](#page-13-3) [2024,](#page-13-3) [Zhang et al.,](#page-13-4) [2024,](#page-13-4) [Singhania](#page-13-5) [et al.,](#page-13-5) [2024\]](#page-13-5) select individual tokens per query without learning, while Focus learns group structure across the entire sequence. The approaches are complementary.

LoRA [\[Hu et al.,](#page-11-2) [2022\]](#page-11-2) is the dominant parameter-efficient adaptation method (see also DoRA [\[Liu et al.,](#page-12-2) [2024b\]](#page-12-2)). We compare in Section [3.3.](#page-5-0)

