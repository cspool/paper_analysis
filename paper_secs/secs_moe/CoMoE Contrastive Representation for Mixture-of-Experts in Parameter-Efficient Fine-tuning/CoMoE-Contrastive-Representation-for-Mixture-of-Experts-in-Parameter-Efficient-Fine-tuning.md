# CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

Jinyuan Feng!2\* Chaopeng Wei®\* Tenghai Qiu!' Tianyi Hu!" Zhiqiang Pu!" ' Institute of Automation, Chinese Academy of Sciences ? School of Artificial Intelligence, University of Chinese Academy of Sciences 3 University of Science and Technology Beijing

#### Abstract

In parameter-efficient fine-tuning, mixtureof-experts (MoE), which involves specializing functionalities into different experts and sparsely activating them appropriately, has been widely adopted as a promising approach to trade-off between model capacity and computation overhead. However, current MoE variants fall short on heterogeneous datasets, ignoring the fact that experts may learn similar knowledge, resulting in the underutilization of MoE's capacity. In this paper, we propose Contrastive Representation for MoE (CoMogB), a novel method to promote modularization and specialization in MoE, where the experts are trained along with a contrastive objective by sampling from activated and inactivated experts in top-k routing. We demonstrate that such a contrastive objective recovers the mutualinformation gap between inputs and the two types of experts. Experiments on several benchmarks and in multi-task settings demonstrate that CoMoE can consistently enhance MoE's capacity and promote modularization among the experts.

### 1 Introduction

Parameter-Efficient Fine-Tuning (PEFT) has emerged to efficiently adapt Large Language Models (LLMs) to downstream tasks by updating only a subset of parameters, significantly reducing computational and memory overhead (Hu et al., 2022a; Liu et al., 2022; He et al., 2021). However, it struggles with substantially increased dataset sizes, especially heterogeneous training datasets, which poses a significant practical challenge (Huang et al., 2024; Wang et al., 2024b). Mixture-of-Experts (MoE) offers a versatile solution to the challenge for its modular design (Zhang et al., 2024).

Thus, Low-rank Adaptation (LoRA), as a popular and effective PEFT method, has been widely integrated with MoE (Dou et al., 2024; Li et al., 2024), leveraging MoE's modularity to enhance the model's capacity and performance. By sparsely activating a subset of experts, LoRA's MoE variants achieve efficient training on heterogeneous datasets and allocate the experts adaptively (Tian et al., 2024). Specifically, the sparse activation is controlled through a router mechanism (e.g., topk routing) that dispatches inputs to the activated experts. Basically, given an input token, only a subset of specialized experts contribute to the output, while other irrelevant experts remain inactive.

Ideally, each expert should specialize in distinct representation subspaces and semantic skills, thereby collaboratively enhancing the model's representational capacity and enabling a broader spectrum of knowledge (Liu et al., 2023). However, despite the explicit division into multiple experts in MoE architecture, its modularization degree remains questionable. Two issues persist: (1) expert knowledge redundancy, where insufficient specialization constraints lead to overlapping functionalities among experts, limiting the model's capacity (Feng et al., 2025); (2) expert load imbalance, where inadequate modularity and specialization during training result in frequent activation of only a subset of experts, which underutilizes other experts and contradicts its original design intent. Consequently, as some studies have indicated (Qian et al., 2024), simply stacking more experts does not linearly improve performance; instead, it leads to a performance bottleneck. Existing studies propose load balance loss (Li et al., 2024) and localized balancing constraint (Dou et al., 2024) to alleviate the mentioned issues, but that is still far from enough.

In this paper, we propose a novel perspective to promote the specialization of experts. As illustrated in Fig. 1, building upon top-k routing, we categorize the experts into activated experts and inactivated experts. Then, we quantify the specialization of experts by mutual information (MI)

<sup>&</sup>quot;Equal contributions.

t Corresponding author.

![](_page_1_Figure_0.jpeg)

Figure 1: Given an input token 2, (a) illustrates a workflow of top-2 routing, which serves as a fundamental mechanism of CoMoE; (b) illustrates the motivation of CoMoE: maximizing MI between input zx and activated experts while minimizing MI between input x and inactive experts.

between the input token and the two types of experts. To promote expert specialization, we define an MI gap, which is derived from the aforementioned MI, and aim to maximize it. In practice, based on the InfoNCE theory (Oord et al., 2018), such an MI gap can be approximated via a contrastive objective by using positive samples from the activated experts and negative samples from the inactive experts (Lan et al., 2024; Wen et al., 2024). The contrastive objective is incorporated as an auxiliary objective during training, encouraging specialization and modularization among experts. We name the proposed method Contrastive Representation for MoE (CoMOogB), a novel MoE variant. Empirically, we evaluate CoMoE on diverse benchmarks, showcasing its remarkable performance on heterogeneous tasks. Summary of our contributions:

- ¢ We define an MI gap to quantify expert specialization and redundancy in top-k routing, with contrastive learning providing an efficient estimation approach.
- ¢ We propose a novel MoE variant, named Co-MoE, which incorporates an auxiliary contrastive objective to enhance expert specialization and modularization.
- ¢ Comprehensive experiments are conducted to demonstrate that our method consistently improves MoE on heterogeneous tasks.

#### 2 Preliminaries

LoRA Basics LoRA (Huet al., 2022a) introduces a pair of low-rank matrices A and B to reparameterize the pretrained weights Wo in a residual manner. Specifically, input x is processed through both the frozen weights and the low-rank matrices:

$$y' = \mathbf{W_0}x + \mathbf{BA}x,\tag{1}$$

where y' denotes the output, with A € R'\*® and B € R&\*". The rank r < min(dj, dz) is significantly small to reduce tunable parameters.

Mixture of Experts In LoRA's MoE variants, the original LORA module is substituted with n parallel experts, each denoted as {£;(%) = ByAyx}"\_,. These experts are activated via a router g(x; G) to process the input collaboratively. Specifically, given an input «, the router calculates the importance of each expert, and the output y' is computed residually as a weighted sum of outputs from the experts:

$$y' = \mathbf{W_0}x + \sum_{i=1}^{n} g_i(x; \mathbf{G}) E_i(x),$$
 (2)

where gi(z;G) represents the weight of the 7-th expert, and £;(x) denotes the output of expert 7.

Top-k Routing Top-k routing is a common and effective routing strategy of the router g(x; G) in MoE, which sparsely activates a subset of the experts. Specifically, only the top & experts with the highest values in g(a; G) are activated. Then, g(x; G) is renormalized for the activated experts.

The renormalization is computed as follows:

$$\hat{g}_i(x) = \begin{cases} \frac{g_i(x)}{\sum_{j \in \text{top}(g(x), k)} g_j(x)} & \text{if } i \in \text{top}(g(x), k) \\ 0 & \text{if } i \notin \text{top}(g(x), k) \end{cases},$$
(3)

where top(g(x), &) returns the indices of the largest k elements in g(x).

