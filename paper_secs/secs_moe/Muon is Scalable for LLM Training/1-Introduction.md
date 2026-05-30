# 1 Introduction

The rapid advancement of large language models (LLMs) (OpenAI et al. 2024; DeepSeek-AI et al. 2024; Grattafiori et al. 2024; Gemini Team et al. 2024) has significantly pushed forward the progress in artificial general intelligence. However, training capable LLMs remains a computationally intensive and resource-demanding process due to scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022). Optimizers play a crucial role in efficiently and effectively training of LLMs, with Adam (Kingma et al. 2015) and its variant AdamW (Loshchilov et al. 2019) being the standard choice for most large-scale training.

Recent developments in optimization algorithms have shown potential to improve training efficiency beyond AdamW (Liu et al. 2024; K. Jordan et al. 2024; Yuan et al. 2024; Vyas et al. 2025; X.-L. Li 2018a; X.-L. Li 2018b; Pooladzandi et al. 2024; X. Li 2022; X.-L. Li 2024; Pethick et al. 2025). Among these, K. Jordan et al. 2024 proposed Muon, which updates matrix parameters with orthogonalized gradient momentum using Newton-Schulz iteration. Initial experiments with Muon have demonstrated promising results in small-scale language model training. However, as discussed in this blog (K. Jordan et al. 2024), several critical challenges remain unaddressed: (1) how to effectively scale optimizers based on matrix orthogonalization to larger models with billions of parameters trained with trillions of tokens, (2) how to compute approximate orthogonalization in a distributed setting, and (3) whether such optimizers can generalize across different training stages including pre-training and supervised finetuning (SFT).

In this technical report, we present a comprehensive study addressing these challenges. Our work builds upon Muon while systematically identifying and resolving its limitations in large-scale training scenarios. Our technical contributions include:

- Analysis for Effective Scaling of Muon: Through extensive analysis, we identify that weight decay plays a crucial
  role in Muon's scalability. Besides, we propose scale adjustments to Muon's parameter-wise update rule. Such
  adjustments allow Muon to work out-of-the-box without hyper-parameter tuning, and also significantly improve
  training stability.
- Efficient Distributed Implementation: We develop a distributed version of Muon with ZeRO-1 (Rajbhandari et al. 2020) style optimization, achieving optimal memory efficiency and reduced communication overhead while preserving the mathematical properties of the algorithm.
- Scaling Law Validation: We performed scaling law research that compares Muon with strong AdamW baselines, and showed the superior performance of Muon (1a). Based on the scaling law results, Muon achieves comparable performance to AdamW trained counterparts while requiring only approximately 52% of the training FLOPs.

Our comprehensive experiments demonstrate that Muon can effectively replace AdamW as the de facto optimizer for large-scale LLM training, offering significant improvements in both training efficiency and model performance. As a result of this work, we release Moonlight, a 16B-parameter MoE model trained using Muon, along with our implementation and intermediate training checkpoints to facilitate further research in scalable optimization techniques for LLMs.

#### 2 Methods

#### 2.1 Background

<span id="page-1-1"></span>The Muon Optimizer Muon (K. Jordan et al. 2024) has recently been proposed to optimize neural network weights representable as matrices. At iteration t, given current weight  $\mathbf{W}_{t-1}$ , momentum  $\mu$ , learning rate  $\eta_t$  and objective  $\mathcal{L}_t$ , the update rule of the Muon optimizer can be stated as follows:

<span id="page-1-0"></span>
$$\mathbf{M}_{t} = \mu \mathbf{M}_{t-1} + \nabla \mathcal{L}_{t}(\mathbf{W}_{t-1})$$

$$\mathbf{O}_{t} = \text{Newton-Schulz}(\mathbf{M}_{t})^{1}$$

$$\mathbf{W}_{t} = \mathbf{W}_{t-1} - \eta_{t} \mathbf{O}_{t}$$
(1)

Here,  $\mathbf{M}_t$  is the momentum of gradient at iteration t, set as a zero matrix when t=0. In Equation 1, a Newton-Schulz iteration process (Bernstein et al. 2024) is adopted to approximately solve  $(\mathbf{M}_t \mathbf{M}_t^{\mathrm{T}})^{-1/2} \mathbf{M}_t$ . Let  $\mathbf{U} \mathbf{\Sigma} \mathbf{V}^{\mathrm{T}} = \mathbf{M}_t$  be the singular value decomposition (SVD) of  $\mathbf{M}_t$ , we will have  $(\mathbf{M}_t \mathbf{M}_t^{\mathrm{T}})^{-1/2} \mathbf{M}_t = \mathbf{U} \mathbf{V}^{\mathrm{T}}$ , which orthogonalizes  $\mathbf{M}_t$ . Intuitively, orthogonalization can ensure that the update matrices are isomorphic, preventing the weight from learning along a few dominant directions (K. Jordan et al. 2024).

<sup>&</sup>lt;sup>1</sup>In practice, we follow (K. Jordan et al. 2024) to use a Nesterov-style momentum by putting  $\mu \mathbf{M}_t + \nabla \mathcal{L}_t(\mathbf{W}_{t-1})$  to the Newton-Schulz iteration instead of  $\mathbf{M}_t$ .

Newton-Schulz Iterations for Matrix Orthogonalization Equation [1](#page-1-0) is calculated in an iterative process. At the beginning, we set X<sup>0</sup> = Mt/∥Mt∥F. Then, at each iteration k, we update X<sup>k</sup> from Xk−<sup>1</sup> as follows:

<span id="page-2-0"></span>
$$\mathbf{X}_{k} = a\mathbf{X}_{k-1} + b(\mathbf{X}_{k-1}\mathbf{X}_{k-1}^{\mathrm{T}})\mathbf{X}_{k-1} + c(\mathbf{X}_{k-1}\mathbf{X}_{k-1}^{\mathrm{T}})^{2}\mathbf{X}_{k-1}$$
(2)

where X<sup>N</sup> is the result of such process after N iteration steps. Here a, b, c are coefficients. In order to ensure the correct convergence of Equation [2,](#page-2-0) we need to tune the coefficients so that the polynomial f(x) = ax + bx<sup>3</sup> + cx<sup>5</sup> has a fixed point near 1. In the original design of K. Jordan et al. [2024,](#page-11-0) the coefficients are set to a = 3.4445, b = −4.7750, c = 2.0315 in order to make the iterative process converge faster for small initial singular values. In this work, we follow the same setting of coefficients.

Steepest Descent Under Norm Constraints Bernstein et al. [2024](#page-11-12) proposed to view the optimization process in deep learning as steepest descent under norm constraints. From this perspective, we can view the difference between Muon and Adam (Kingma et al. [2015;](#page-11-5) Loshchilov et al. [2019\)](#page-11-6) as the difference in norm constraints. Whereas Adam is a steepest descent under the a norm constraint dynamically adjusted from a Max-of-Max norm, Muon offers a norm constraint that lies in a static range of Schatten-p norm for some large p (Franz [2024\)](#page-11-13). When equation [1](#page-1-0) is accurately computed, the norm constraint offered by Muon will be the spectral norm. Weights of neural networks are used as operators on the input space or the hidden space, which are usually (locally) Euclidean (Cesista [2024\)](#page-11-14), so the norm constraint on weights should be an induced operator norm (or spectral norm for weight matrices). In this sense, the norm constraint offered by Muon is more reasonable than that offered by Adam.

