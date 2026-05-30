# A Appendix

#### <span id="page-13-2"></span>A.1 Loss-Free Load Balancing Combination

<span id="page-13-3"></span>Table 8: Model setup and performance. Sequence-wise Expert Utilization (SEU) is computed as the mean over the fraction of activated experts within a sequence. SIMBAL can improve sequence-wise balance without significant performance degradation, sometimes improving performance. All models use all experts throughout the full validation set, LF is the least balanced per-batch. While LBL asserts near-perfect balance, it also causes substantial perplexity degradation.

| Model        | MoE-M   | MoE-M   | MoE-M     | MoE-M   | MoE-M   | MoE-M     |
|--------------|---------|---------|-----------|---------|---------|-----------|
| Gating       | Softmax | Softmax | Softmax   | Sigmoid | Sigmoid | Sigmoid   |
| Balancing    | LF      | LF+LBL  | LF+SimBal | LF      | LF+LBL  | LF+SimBal |
| Perplexity Ó | 13.708  | 14.154  | 13.695    | 13.618  | 14.015  | 13.637    |
| SEU Ò        | 0.505   | 0.997   | 0.755     | 0.381   | 0.997   | 0.476     |

Loss-Free (LF) balancing [\[Wang et al., 2024\]](#page-12-5) applies a direct bias to routing scores (s " xR, rather than routing weights r " GpxRq) without adding an auxiliary loss. Let f<sup>i</sup> be the expert frequency in the current batch and ¯f " 1{E the uniform target. Each expert's score is adjusted by a fixed scalar γ:

$$b_i' = b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i) \tag{7}$$

The scores are then used for computing the top-A experts with the new scores s<sup>i</sup> :

$$s_i = xR + b_i \tag{8}$$

This encourages uniform expert assignment, but is not used in the weighting of the experts (r). It thus allows non-uniform expert weighting but still allocates experts uniformly over the full dataset. Additionally, γ is a hyperparameter that may need to be tuned, though the original authors recommend 0.001 since it provides a good balance between balancing while preventing fluctuations later in training.

Other work [\[DeepSeek-AI et al., 2025\]](#page-10-4) use LBL in conjunction with LF for batch-wise load balancing, as they find that it can result in substantial imbalance in expert use sequence-wise. We do not include these results in earlier charts due to this extreme imbalance. Instead, in this section, we explore whether a combination with SIMBAL works similarly to LBL to improve sequence-wise balancing.

While the original authors of LF use sigmoid gating (over our softmax gating), we find that softmax gating is substantially more common in state-of-the-art work. Thus, to maximize relevance (regardless of performance), we additionally compare with softmax gating. The training setup for MoE-M remains identical to Section [3.2](#page-3-1) otherwise.

We evaluated the balancing capabilities of this method using the MoE-M configuration, comparing its performance against both LBL and SIMBAL. We summarize our results in Table [8.](#page-13-3) We find that sigmoid gating leads to significant degradation in sequence-wise balance, especially compared to using only SIMBAL or LBL (as seen in Table [4\)](#page-7-0). In exchange, there was a minor and possibly statistically insignificant (using the deviation values from Section [4.2.](#page-5-0) This is not ideal, as with larger models, when using model parallelism, extra consideration may be needed to ensure full utilization of all devices. Using LBL mitigates some of this, but leads to a substantial degradation in performance.

#### <span id="page-13-1"></span>A.2 Layer-Wise Orthogonalization

We provide tables for layer-wise orthogonalization performance for SIMBAL, and compare the results to LBL on MoE-M (Table [9\)](#page-14-0) and MoE-L (Table [10\)](#page-14-1). LBL alone does not orthogonalize the router whatsoever, while SIMBAL is able to achieve mean squared error similar to commonly used ϵ for numerical stability.

