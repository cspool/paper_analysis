# 7 DISCUSSION

### 7.1 Why the Method Works

The effectiveness of pre-attention prediction stems from the fundamental information flow within transformer architectures [\(Abnar & Zuidema,](#page-10-0) [2020\)](#page-10-0). Pre-attention activation tensors represent the token state immediately before expert routing decisions, providing the more relevant and temporally accurate information for predicting routing outcomes.

Unlike cross-layer approaches that attempt to extrapolate routing decisions from previous layer states, our method leverages the input closer to the original routing function. Using the tensor in the same layer eliminates the uncertainty inherent in temporal prediction across layer boundaries and provides access to the complete information set used by the routing mechanism.

The superior performance on simpler models like Phi-mini (97.62% accuracy) compared to more complex models (93.03-94.69% accuracy) might reflect the relationship between routing complexity and prediction difficulty. Models with more specialized expert functions and complex routing patterns present greater prediction challenges, but still benefit substantially from the pre-attention information.

Analysis of model-specific patterns reveals that pre-attention activation tensors capture both semantic content and structural patterns that determine expert routing. The activation tensors encode information about token types, positional patterns, and contextual relationships that routing functions use to make expert selection decisions, consistent with recent studies of attention mechanisms in MoE models [\(Piekos](#page-11-0) [et al.,](#page-11-0) [2025;](#page-11-0) [Yang et al.,](#page-11-0) [2025b\)](#page-11-0).

#### 7.2 Future Work

End-to-end system integration building on recent advances in MoE system design represents the most immediate opportunity for extending this work. Combining our prediction approach with optimized caching strategies and dynamic overprovisioning based on inference patterns could further improve performance across diverse deployment scenarios.

Extension to larger MoE models with more experts and more complex routing patterns would validate the approach's scalability. While current results demonstrate effectiveness across models with 2-8 selected experts, larger models with 16+ expert selections might present additional challenges and optimization opportunities.

Dynamic over-provisioning strategies that adjust expert loading based on prediction confidence scores may further optimize the trade-off between I/O overhead and hit rates. Predictions with high confidence scores could operate in exact match mode while less confident predictions could increase over-provisioning to maintain hit rate targets.

