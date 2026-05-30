# **3 Efficiency Leverage: Metric for Quantifying MoE Compute-Efficiency**

Next, we define the efficiency leverage and use it to outline our objectives and roadmap.

*Definition of Efficiency Leverage.* To quantify the computational efficiency gain of MoE compared to dense models, we introduce core metric of **Efficiency Leverage (EL)**. Let XDense denote a standard dense architecture and XMoE represent a MoE architectures. Within XMoE, models share identical core configurations (attention mechanisms, expert count, granularity, shared experts), scaled solely through hidden dimensions (*d*model, *d*ffn, *d*expert) and layer count *n*layer. Formally, we define the EL of XMoE as the ratio of compute budgets required for the dense and MoE models to achieve the same performance level. While model performance can be quantified through loss values, benchmark scores, or task-specific metrics, this study adopts loss as the primary metric.

**Definition 3.1** (Efficiency Leverage)**.** For XMoE achieving minimal loss L(*C*moe; XMoE) at compute budget *C*moe, assuming there exists a compute budget *C*dense such that XDense attains comparable minimal loss L(*C*dense; XDense), we define the efficiency leverage as:

$$EL(\mathcal{X}_{\text{MoE}} \mid \mathcal{X}_{\text{Dense}}; C_{\text{target}}) = \frac{C_{\text{dense}}}{C_{\text{moe}}},$$
s.t. 
$$|\mathcal{L}(C_{\text{moe}}; \mathcal{X}_{\text{MoE}}) - \mathcal{L}(C_{\text{dense}}; \mathcal{X}_{\text{Dense}})| \le \epsilon \quad (\epsilon \to 0)$$
(5)

Here, the minimal loss achievable by an architecture under specific computational constraints represents its performance ceiling at that scale. An EL greater than 1 signifies that the MoE architecture is more computationally efficient than the dense model, achieving the same performance with less compute. Conversely, an EL less than 1 indicates inferior efficiency.

Following established practice [\(Kaplan et al.,](#page-24-3) [2020\)](#page-24-3), we model the relationship between compute (*C*) and loss (L) with a power law: L(*C*; X ) = *α*<sup>X</sup> *C* <sup>−</sup>*β*<sup>X</sup> . This allows us to simplify the EL definition in the compute-optimal training regime [\(Hoffmann et al.,](#page-23-5) [2022\)](#page-23-5) or similar over-training regime [\(Gadre](#page-23-6) [et al.,](#page-23-6) [2024\)](#page-23-6). Given the computational cost *C* = *M* · *D*, the efficiency leverage simplifies under fixed data size *D* to the ratio of model scales: *EL*(XMoE | XDense) ≈ *M*Dense/*M*MoE. This formulation demonstrates that EL quantifies the relative model scale of XMoE compared to XDense in achieving equivalent performance. *In other words, given the model scale of an MoE and its corresponding efficiency leverage, we can directly determine the equivalent dense model scale required to achieve the same performance.*

*Objective and Roadmap.* Existing studies [\(Ludziejewski et al.,](#page-24-1) [2024;](#page-24-1) [Abnar et al.,](#page-22-0) [2025;](#page-22-0) [Clark](#page-23-1) [et al.,](#page-23-1) [2022\)](#page-23-1) indicate that the model capacity of MoE is significantly influenced by architectural configurations. The primary objective of this work is to understand and quantify how MoE architectural choices influence Efficiency Leverage. Our central research question is:

*How do the architectural configurations of an MoE model affect its Efficiency Leverage, and how does this relationship scale with the computational budget?*

Specifically, our investigation focuses on three critical architectural dimensions[1](#page-7-0) : the *Activation Ratio* (*A*), *Expert Granularity* (*G*), and *Shared Expert Ratio* (*S*). They jointly determine the effective capacity of MoE models, and can be used to derive other MoE configurations (*e.g.,* the number of experts, the number of actived experts) based on the definitions in Section [2.1.](#page-2-0) Our goal is to find the configuration (*A* opt , *G* opt , *S* opt) that maximizes EL for a given compute budget *C*:

$$(A^{\text{opt}}, G^{\text{opt}}, S^{\text{opt}}) = \arg \max_{(A,G,S) \in \mathcal{X}_{\text{MoE}}} EL(\mathcal{X}_{\text{MoE}} \mid \mathcal{X}_{\text{Dense}}; C)$$
(6)

To make the analysis tractable, we assume the effects of these dimensions are largely independent and conduct systematic ablation studies. We start with a baseline MoE architecture (2 of 64 experts activated, plus one shared expert) and vary one dimension at a time across a range of compute budgets (from 3*e*18 to 3*e*20 FLOPs). To ensure a fair and robust comparison, we leverage the findings from our preliminary studies (Sections [2.2](#page-3-0) and [2.3\)](#page-5-1). For each architecture and compute budget, we determine the reasonable model size (*M*) and data size (*D*) using our derived allocation laws and configure training with optimal hyperparameters from our hyperparameter scaling laws. This rigorous protocol ensures that each architecture is evaluated at or near its peak potential for a given budget, yielding reliable and cost-effective conclusions. Further details on the experimental setup are provided in Appendix [B.](#page-26-1) Next, we first empirically analyze the impact of each dimension on EL, and then integrate our empirical findings into a unified scaling law to models the relationship between MoE configurations and the resulting EL.

