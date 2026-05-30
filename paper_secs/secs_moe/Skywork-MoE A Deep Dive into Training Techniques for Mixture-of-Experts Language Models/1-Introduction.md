# 1 Introduction

Recent advancements in the field of artificial intelligence have seen large language models (LLMs) [\(Ouyang et al.,](#page-9-0) [2022;](#page-9-0) [OpenAI,](#page-9-1) [2023;](#page-9-1) [Bubeck et al.,](#page-8-0) [2023;](#page-8-0) [Anthropic,](#page-8-1) [2024;](#page-8-1) [Tou](#page-9-2)[vron et al.,](#page-9-2) [2023b;](#page-9-2) [Meta-AI,](#page-9-3) [2024;](#page-9-3) [Team,](#page-9-4) [2024;](#page-9-4) [DeepSeek-AI,](#page-8-2) [2024b\)](#page-8-2) revolutionize numerous branches of natural language processing (NLP), encompassing tasks from machine translation to automated summarization. However, the computational demands and associated costs of training and deploying state-of-the-art dense LLMs pose significant challenges, particularly

at the scale of tens or hundreds of billions of parameters. In response to these challenges, sparse models, such as Mixture-of-Experts (MoE), have gained prominence [\(Fedus et al.,](#page-8-3) [2022;](#page-8-3) [Lepikhin et al.,](#page-9-5) [2020;](#page-9-5) [Du et al.,](#page-8-4) [2022;](#page-8-4) [Dai et al.,](#page-8-5) [2024;](#page-8-5) [DeepSeek-AI,](#page-8-2) [2024b\)](#page-8-2). These models offer a more economically viable alternative by distributing computation across various specialized sub-models or "experts", potentially matching or even surpassing the performance of their dense counterparts with a fraction of the resource requirements [\(Artetxe et al.,](#page-8-6) [2022;](#page-8-6) [Rajbhandari et al.,](#page-9-6) [2022;](#page-9-6) [Clark et al.,](#page-8-7) [2022\)](#page-8-7).

In light of these developments, this technical report introduces Skywork-MoE, a highperformance MoE large language model with 146 billion parameters and 16 experts. This model leverages the foundational architecture of our previously developed Skywork-13B model [\(Wei et al.,](#page-9-7) [2023\)](#page-9-7), utilizing its dense checkpoints as the initial setup [\(Komatsuzaki et al.,](#page-8-8) [2023\)](#page-8-8). We conduct experimental analysis on relative benefits of two pivotal strategies in LLM development: upcycling from existing dense models versus initiating training from scratch. Through rigorous evaluation, we provide nuanced insights into how the initial conditions and training budgets influence the effectiveness of these approaches, offering practical guidance on their application. Skywork-MoE embodies the forefront of MoE research by incorporating two novel training techniques: gating logit normalization and adaptive auxiliary loss coefficients. The former aims to enhance the diversification among the experts, while the latter facilitates the tailored adjustment of auxiliary loss coefficients at different layers of the model. Moreover, the training of Skywork-MoE was conducted on a condensed subset of the SkyPile corpus [\(Wei et al.,](#page-9-7) [2023\)](#page-9-7), with subsequent evaluations demonstrating its robust performance

<sup>∗</sup> Email: {forename}.{surname}@kunlun-inc.com

across a diverse array of benchmarks. This report aims to detail these innovations and findings, setting a new benchmark for the efficiency and efficacy of MoE models in large-scale language processing tasks.

#### 2 Preliminaries

Skywork-MoE follows the previous work of Switch Transformer (Fedus et al., 2022), which implement the idea of MoE (Jacobs et al., 1991; Eigen et al., 2014; Shazeer et al., 2017) with transformer architecture (Vaswani et al., 2017).

#### 2.1 MoE for Transformers

In a standard transformer, each layer processes inputs through self-attention mechanisms followed by feed-forward neural networks (FFNs) (Vaswani et al., 2017). The transformer processes every token of the input sequence through the same pathways (i.e., every parameter in the model is active for every input).

In contrast, the MoE architecture modifies the typical transformer by replacing some or all of the FFNs with a mixture-of-experts, where each expert is itself a small FFNs, and the MoE layer houses multiple such experts. The MoE layer increases the capacity of transformer models while maintaining computational efficiency by selectively activating some of the expert networks for each input token. The selection of experts is performed by a gating mechanism, allowing the model to dynamically route tokens to the most relevant experts.

The gating mechanism in consists of a soft-max layer that computes a probability distribution over the available experts for each token. The gate output g for the i-th token with embedding  $x_i$  is given by:

<span id="page-1-1"></span>
$$\operatorname{softmax}(Wx_i + b) = (g_{i1}, \dots, g_{in})^T \quad (1)$$

where W is the gating weight matrix, b is the gating bias vector,  $g_{ij}$  is the gating probability of the i-th token being assigned to the j-th expert and n is the total number of experts. The k experts with the highest probability are then selected to process the token, which is also known as top-k routing. Conventionally one chooses k = 1 or k = 2. In this work, we always assume using top-2 routing of experts.

Let's denote the set of selected experts for the *i*-th token as  $\mathcal{E}_i$ . Each selected expert  $j \in \mathcal{E}_i$  processes the token embedding  $x_i$  and generates an output  $\operatorname{Expert}_j(x_i)$ . The outputs from the k selected experts are then linearly combined according to the corresponding gating probabilities:

$$y_i = \frac{1}{s_i} \sum_{j \in \mathcal{E}_i} g_{ij} \cdot \text{Expert}_j(x_i).$$
 (2)

where  $s_i = \sum_{j \in \mathcal{E}_i} g_{ij}$ . The combined output  $y_i$  is then passed to the next layer of the model.

#### 2.2 Auxiliary Loss

To ensure balanced load across experts and prevent a single expert from dominating, Switch Transformer employs an auxiliary loss function that encourages the even distribution of tokens among experts. Let  $p_j$  be the proportions of tokens assigned to expert j. The load is balanced across experts if  $p_j = k/n$  for all  $j = 1, \ldots, n$ . An naive auxiliary loss  $\mathcal{L}_{\text{aux}}$  that directly penalizes the discrepancy between  $p_j$  and k/n would be

<span id="page-1-0"></span>
$$\mathcal{L}_{\text{aux}} = \sum_{j=1}^{n} \left( \frac{k}{n} - p_j \right)^2.$$
 (3)

However, as  $p_j$  is only a statistic that does not allow for back-propagation, the naive auxiliary loss is not applicable in practice. As a differentiable surrogate, one can assume that

$$p_j \approx k \cdot E[g_j] \approx \frac{k}{T} \sum_{i=1}^{T} g_{ij}$$

where T is the number of tokens in a batch. Substituting  $p_j$  by  $\frac{k}{T} \sum_{i=1}^{T} g_{ij}$  in (3), and ignoring the constant k, we obtain

<span id="page-1-2"></span>
$$\mathcal{L}_{\text{aux}} = \sum_{j=1}^{n} \left( \frac{1}{n} - \frac{1}{T} \sum_{i=1}^{T} g_{ij} \right)^{2}, \quad (4)$$

which is the actual auxiliary loss that is commonly used in switch transformer training. By minimizing this loss, the model can effectively learns to balance the load across experts, preventing any single expert from being overloaded or underutilized.

The total loss function  $\mathcal{L}_{total}$  for training the Switch Transformer is a combination of the cross entropy loss  $\mathcal{L}_{ce}$  for the next token prediction task and the auxiliary loss  $\mathcal{L}_{aux}$ , weighted by a hyperparameter  $\alpha$ :

<span id="page-2-0"></span>
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ce}} + \alpha \mathcal{L}_{\text{aux}} \tag{5}$$

By incorporating the MoE layer and the auxiliary loss for load balancing, Switch Transformer enables the efficient scaling of transformer models to billions of parameters while maintaining computational tractability.

