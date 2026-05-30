# 6 PERFORMANCE ANALYSIS

#### 6.1 Accuracy vs. System Complexity

Our same-layer prediction approach achieves substantial accuracy improvements while reducing system complexity compared to cross-layer methods. The 15-19 percentage point improvement over Fate [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0) (93.03% vs 78.79% for DeepSeek-V2) demonstrates that leveraging same-layer information provides superior prediction signals compared to cross-layer approaches. Compared with DuoServe-MoE [\(Zhang et al.,](#page-11-0) [2025b\)](#page-11-0) that achieves 54- 67% top-2 accuracy on Mixtral-8x7B and Mixtral-8x22B, our method generalizes to three MoE models with a much higher 93-98% (+30%) exact-match accuracy. Compared with caching based method HOBBIT [\(Tang et al.,](#page-11-0) [2024\)](#page-11-0) with around 55% hit rate on Mixtral-8x7B and Phi-MoE, we achieve about 40% improvement in absolute accuracy.

The elimination of cross-layer communication overhead represents a significant system simplification. While Fate requires coordinating information across transformer layers and managing temporal dependencies between layer executions, our approach operates independently within each layer using readily available pre-attention activation tensors.

Prediction latency analysis shows minimal overhead with 0.15 ms for expert prediction across all model configurations. This overhead represents less than 10% of the

pre-MLP computation pipeline (pre-attention norm + selfattention + post-attention norm), ensuring that the prediction computation remains sufficiently fast to enable parallel fetching of experts while the current layer executes, which preserves the parallelization benefits of expert prediction.

#### 6.2 I/O Savings Analysis

The substantial accuracy improvements translate directly into quantifiable I/O performance benefits. Our improved prediction accuracy directly reduces the frequency of expert loading operations during inference. When predictions are correct (93.03% of tokens), experts are prefetched during self-attention computation, achieving zero loading latency when the pipeline reaches expert selection. When predictions fail (6.97% of tokens), the system loads experts normally without additional penalty.

The expected expert loading time per token is calculated as the misprediction rate multiplied by the expert loading time. Our approach achieves expected loading times of 0.27-0.64 ms/token compared to Fate's 0.85-2.01 ms/token across different hardware configurations. On Tesla V100 systems, our expected time is (1 − 0.9303) × 9.5 = 0.66 ms/token compared to Fate's (1 − 0.7879) × 9.5 = 2.01 ms/token, providing 1.37 ms savings per token.

The frequency-based improvements are substantial: 93.03% of tokens experience zero expert loading latency compared to 78.79% with Fate's approach, representing 14.24 percentage points more tokens with immediate expert access. Over extended inference sessions of 1000 tokens, this translates to 569-1352 ms total latency savings across different hardware configurations, excluding the reloading time on wrong predictions.

Overprovisioning strategies further improve performance by loading 10 experts instead of the required 6, achieving 98.65% prediction accuracy. This reduces expected loading time to (1−0.9865)×loading time = 1.35%×loading time, making 98.65% of tokens achieve zero loading latency at the cost of 67% additional I/O overhead during prefetching.

### 6.3 Deployment Strategy Recommendations

Cloud environment with enough resources should adopt overprovisioning strategies that load 10 experts instead of required 6 to achieve 98 + % hit rates. The 67% I/O overhead increase might be accommodated by cloud-scale I/O bandwidth [\(Bodner et al.,](#page-10-0) [2025\)](#page-10-0) while the < 2% miss rate minimizes performance-critical cache miss penalties.

Edge devices with I/O bandwidth constraints should utilize strategies based on their parallel loading capacity. For devices that can load multiple experts in parallel with attention computation, precise prediction with 93.03% accuracy provides optimal resource utilization. However, for edge

<span id="page-9-0"></span>scenarios where I/O bandwidth limits parallel loading to a single expert during the attention computation window, top-1 prediction becomes critical. Our top-1 prediction achieves 98.6-99.1% hit rates across the different models, ensuring that loading the highest-confidence predicted expert in parallel with attention processing maximizes the utility of the limited parallel execution window.

