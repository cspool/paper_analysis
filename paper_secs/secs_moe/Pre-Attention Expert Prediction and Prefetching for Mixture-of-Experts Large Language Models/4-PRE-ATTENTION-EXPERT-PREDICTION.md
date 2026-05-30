# 4 PRE-ATTENTION EXPERT PREDICTION

### 4.1 Problem Formulation

We formulate the general expert prediction problem for different MoE configurations and deployment scenarios. For example, Qwen3-30B selects 8 experts, but Phi-mini selects 2 experts in one FFN. Thus, we predict the top-k expert indices that will be selected by the original routers.

Let X ∈ R<sup>d</sup> denote the pre-attention weights for a given token at layer l, where d represents the hidden dimension. The standard MoE routing mechanism computes expert selection through equations (1)-(2), where W<sup>g</sup> ∈ R<sup>E</sup>×<sup>d</sup> are the gating parameters for E experts, and TopK(·, k) selects the indices of the k highest-scoring experts.

$$\mathbf{g} = \text{Softmax}(\mathbf{W}_g \cdot \mathbf{X}) \tag{1}$$

$$\hat{\mathbf{Y}}_{\text{true}} = \text{TopK}(\mathbf{g}, k) \tag{2}$$

Our objective is to learn a mapping f(X; θ) → Yˆ that accurately predicts the expert selection Yˆ = {y1, y2, . . . , yk} using only the pre-attention information available within the current layer. As equations (3)-(4) show, s ∈ R<sup>E</sup> represents the predicted expert selection scores, and we select the top-k experts by ranking these scores directly.

$$\mathbf{s} = f(\mathbf{X}; \ \theta) \tag{3}$$

$$\hat{\mathbf{Y}} = \text{TopK}(\mathbf{s}, k) \tag{4}$$

We address three distinct deployment scenarios through different formulations. For standard deployment scenarios, we predict the precise set of k experts that will be selected. For over-provisioning scenarios, we predict a larger set of experts (e.g., 10 instead of 6 for DeepSeek V2 Lite) to achieve higher hit rates at the cost of increased I/O overhead. For I/O bandwidth-constrained edge scenarios where only one expert can be loaded in parallel with attention computation, we evaluate top-1 accuracy, which measures whether the single highest-scoring predicted expert is among the k experts that will actually be selected by the routing function.

#### 4.2 Pre-Attention Prediction Workflow

Our approach exploits the natural pipeline timing of transformer inference to perform expert prediction with minimal overhead. During the standard transformer forward pass, pre-attention normalization produces weights that capture the token representation immediately before expert routing.

| Hardware Configuration | Timing (ms)      |
|------------------------|------------------|
| Pre-Attention Norm     |                  |
| Tesla V100-SXM2-32GB   | 0.1292 ± 0.0120  |
| NVIDIA A100-PCIE-40GB  | 0.0771 ± 0.0007  |
| NVIDIA A100 80GB PCIe  | 0.0750 ± 0.0015  |
| Self-Attention         |                  |
| Tesla V100-SXM2-32GB   | 1.1279 ± 0.0388  |
| NVIDIA A100-PCIE-40GB  | 0.7607 ± 0.0069  |
| NVIDIA A100 80GB PCIe  | 0.7385 ± 0.0074  |
| Post-Attention Norm    |                  |
| Tesla V100-SXM2-32GB   | 0.1292 ± 0.0120  |
| NVIDIA A100-PCIE-40GB  | 0.0823 ± 0.0012  |
| NVIDIA A100 80GB PCIe  | 0.0797 ± 0.0040  |
| Expert Selection       |                  |
| Tesla V100-SXM2-32GB   | 0.1432 ± 0.0138  |
| NVIDIA A100-PCIE-40GB  | 0.0972 ± 0.0025  |
| NVIDIA A100 80GB PCIe  | 0.1018 ± 0.0039  |
| Expert Computation     |                  |
| Tesla V100-SXM2-32GB   | 10.3075 ± 1.7038 |
| NVIDIA A100-PCIE-40GB  | 6.1970 ± 1.0668  |
| NVIDIA A100 80GB PCIe  | 6.8111 ± 1.1864  |

Table 1. Timing comparison of key transformer operations in DeepSeek-V2-Lite across different GPU configurations. All measurements represent mean ± standard deviation over 50 samples.

![](_page_4_Figure_1.jpeg)

Figure 4. Expert Selector Architecture Comparison

We clone these weights to the CPU for parallel prediction computation while the GPU continues with self-attention.

The prediction process operates in three stages. First, we extract the pre-attention weights X immediately after layer normalization and before self-attention computation. Second, we feed these weights through our trained prediction model fl(X; θl) to generate expert probability scores, where we maintain a separate predictor for each layer l with layerspecific parameters θ<sup>l</sup> . This layer-wise approach allows each predictor to specialize in the unique expert selection patterns characteristic of its corresponding transformer layer. Third, we select the top-k experts based on these scores for prefetching or caching decisions. The critical advantage of this approach is timing. The prediction computation occurs in parallel with self-attention, which typically requires 0.73-1.13 milliseconds across different hardware configurations (Table [1\)](#page-3-0). This parallel execution window provides sufficient time for both prediction computation (0.075-0.129 ms) and an early start for expert prefetching operations without introducing additional latency to the inference pipeline.

