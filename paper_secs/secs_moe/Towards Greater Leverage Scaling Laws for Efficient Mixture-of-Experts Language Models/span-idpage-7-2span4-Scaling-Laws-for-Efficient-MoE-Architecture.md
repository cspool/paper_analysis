# <span id="page-7-2"></span>**4 Scaling Laws for Efficient MoE Architecture**

To achieve greater leverage, we first conduct an extensive empirical study on the architectural configurations of MoE and derive scaling laws for efficient MoE architectures.

#### <span id="page-7-1"></span>**4.1 Empirical Study on the Interplay between Efficiency Leverage and MoE Architecture**

To identify the MoE architecture that maximizes Efficiency Leverage (EL) for a given compute budget, we systematically investigate the impact of several key design choices. These include

<span id="page-7-0"></span><sup>1</sup>Other architectural configurations, such as the arrangement of MoE and dense layers, have been verified to have a secondary impact on the efficiency leverage of MoE. See Appendix [B](#page-26-1) and Appendix [D](#page-28-0) for details.

the activation ratio, expert granularity, shared expert ratio, and other configurations. For each architectural dimension, we vary it systematically while holding other factors and the model scale *M* constant. To ensure a fair comparison, all models are trained following the configurations derived from our scaling laws (Section [2\)](#page-2-1), which specify the ideal model size (*M*), data volume (*D*), and hyperparameters for any given total compute budget. Guided by the scaling laws for optimal model-data allocation (defined in Section [2.3,](#page-5-1) we train each model on over three times its optimal number of tokens. This was done to simulate the overtrained state commonly observed in real-world scenarios. All of trained models can be found in Appendix [F.](#page-30-0) Based on the observed training dynamics, we plot the resulting loss curves and EL trends to isolate and quantify the influence of each design choice. To ensure robust analysis, we presuppose a standard power-law relationship between FLOPs cost and training loss, and observe the loss of experimental models after sufficient training using the theoretically optimal allocation as a reference.

#### **4.1.1 Optimal Expert Activation Ratio**

We begin by investigating the activation ratio (*A*), a critical factor governing MoE efficiency. Our experimental design isolates the effect of *A* by holding the computational cost per token (*M*) constant. This is achieved by fixing the number of activated experts and their granularity, while varying the total number of experts in the pool from 2 to 256. This setup allows us to explore a wide range of activation ratios (from 0.8% to 100%, where 100% represents a dense model) without altering the forward pass FLOPs. The optimization problem for a given compute budget *C* is thus:

$$A^{\text{opt}} = \arg\min_{A} \mathcal{L}(A; C, M, G, S)$$
 (7)

The IsoFLOPs curves, presented in Figure [5a,](#page-9-0) reveal a clear and consistent trend. Across all tested FLOPs budgets (from 1*e*18 to 3*e*20), loss monotonically decreases with activation ratio, following a power-law pattern. For all configurations, the lowest tested ratio of 0.8% consistently yields the minimum loss. This finding suggests a core principle: for a fixed computational cost, greater model sparsity (*i.e.,* lower activation ratio) leads to higher parameter efficiency.

To quantify this efficiency improvement, we fit a series of loss scaling curves at different activation ratios. Based on these curves, we compute the efficiency leverage for different activation ratios and FLOPs budgets, as illustrated in Figure [5b.](#page-9-0) The results reveal two key trends. First, for a fixed FLOPs budget, the EL consistently increases as the activation ratio decreases, indicating that sparse activation can always enhance computational efficiency. Second, for a fixed activation ratio, the EL grows with the computational budget, demonstrating that the MoE advantage is amplified at larger scales. These findings confirm that reducing the activation ratio yields substantial efficiency gains, and these benefits are magnified in large-scale, high-computation regimes.

#### **Key Takeaway 1**

- **Monotonic Relationship Between Efficiency and Activation Ratio.** For a fixed computational cost, model performance consistently improves as the activation ratio decreases. This indicates a direct, monotonic relationship between sparsity and efficiency.
- **Efficiency Gains Amplify with Scale.** The efficiency advantage of MoE models (their EL) grows with the total training budget. This highlights their suitability for large-scale training, where their benefits become even more significant.

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 5 **Impact of the Activation Ratio** *A* **on Loss and Efficiency.** (a) At any fixed compute budget (each colored line), lower activation ratios yield lower loss. The orange stars mark the optimal (lowest) loss point. (b) Loss and EL scaling curves illustrate that EL increases with both higher compute budgets and lower activation ratios, showing that MoE advantages are magnified at scale.

