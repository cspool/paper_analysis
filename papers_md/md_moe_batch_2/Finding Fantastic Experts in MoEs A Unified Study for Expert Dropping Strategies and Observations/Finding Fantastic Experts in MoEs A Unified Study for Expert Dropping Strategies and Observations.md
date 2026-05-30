# Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

Ajay Jaiswal <sup>1</sup> Jianyu Wang <sup>2</sup> Yixiao Li <sup>2</sup> Pingzhi Li <sup>3</sup> Tianlong Chen <sup>3</sup> Zhangyang Wang <sup>1</sup> Chong Wang <sup>2</sup> Ruoming Pang <sup>2</sup> Xianzhi Du <sup>2</sup>

### Abstract

Sparsely activated Mixture-of-Experts (SMoE) has shown promise in scaling up the learning capacity of neural networks. However, vanilla SMoEs have issues such as expert redundancy and heavy memory requirements, making them inefficient and non-scalable, especially for resourceconstrained scenarios. Expert-level sparsification of SMoEs involves pruning the least important experts to address these limitations. In this work, we aim to address three questions: 1 What is the best recipe to identify the least knowledgeable subset of experts that can be dropped with minimal impact on performance? 2 How should we perform expert dropping (one-shot or iterative), and what correction measures can we undertake to minimize its drastic impact on SMoE subnetwork capabilities? 3 What capabilities of full-SMoEs are severely impacted by the removal of the least dominant experts, and how can we recover them? *Firstly,* we propose MoE Experts Compression Suite (MC-Suite), which is a collection of some previously explored and multiple novel recipes to provide a comprehensive benchmark for estimating expert importance from diverse perspectives, as well as unveil numerous valuable insights for SMoE experts. *Secondly,* unlike prior works with a one-shot expert pruning approach, we explore the benefits of iterative pruning with the reestimation of the MC-Suite criterion. Moreover, we introduce the benefits of task-agnostic finetuning as a correction mechanism during iterative expert dropping, which we term MoE Lottery Subnetworks. *Lastly,* we present an experimentally validated conjecture that, during expert dropping, SMoEs' instruction-following capabilities are predominantly hurt, which can be restored to

a robust level subject to external augmentation of instruction-following capabilities using k-shot examples and supervised fine-tuning.

### 1. Introduction

Sparsely activated Mixture-of-Experts (SMoEs) are a promising architecture design that facilitates an amalgamation of the collective intelligence of multiple experts and are distinguished by their ability to dynamically allocate computational resources based on the input. Mixture-of-Experts, initially introduced in [\(Shazeer et al.,](#page-12-0) [2017a\)](#page-12-0), has undergone extensive exploration and advancement, and is now adopted in industry-scale LLMs (*e.g.*, Mixtral-8×7B, Grok-1, DBRX, *etc.*), achieving stellar performance across various NLP and CV task leaderboards. Despite the sparse nature of MoEs promising enhanced efficiency and scalability, they have crucial limitations: 1 SMoEs trade space for FLOPs, which require high memory usage due to the duplication of the network layers into multiple copies as experts; 2 SMoEs tend to have poor utilization of their capacity and existence of redundancy [\(Mittal et al.,](#page-11-0) [2022;](#page-11-0) [Chen et al.,](#page-9-0) [2023\)](#page-9-0) due to representation collapse.

In parallel to well-studied techniques that address memory and compute bottlenecks using weight sparsity [\(Jaiswal](#page-11-1) [et al.,](#page-11-1) [2023c;](#page-11-1) [Lee et al.,](#page-11-2) [2019;](#page-11-2) [Frankle & Carbin,](#page-10-0) [2019;](#page-10-0) [Yin](#page-12-1) [et al.,](#page-12-1) [2023b;](#page-12-1) [Liu et al.,](#page-11-3) [2023a\)](#page-11-3) and quantization [\(Liu et al.,](#page-11-4) [2023b;](#page-11-4) [Kim et al.,](#page-11-5) [2023;](#page-11-5) [Dettmers et al.,](#page-10-1) [2023;](#page-10-1) [Frantar et al.,](#page-10-2) [2022;](#page-10-2) [Lin et al.,](#page-11-6) [2023\)](#page-11-6), SMoEs architecture design facilitates a unique opportunity for *expert-level sparsification* that aims to compact the SMoE model by retaining fewer but more knowledgeable experts. For instance, Figure [2](#page-2-0) illustrates that the existence of some experts is *critically important (dominant)* and dropping them could lead to an abrupt performance drop, while some experts are notably redundant with negligible impact when removed. Recently, a few works have proposed expert importance estimation techniques such as token reconstruction loss [\(Lu et al.,](#page-11-7) [2024\)](#page-11-7) and heavy-hitters counting [\(Muzio et al.,](#page-12-2) [2024\)](#page-12-2), illustrating the potential of expert dropping. However, a comprehensive benchmarking of possible task-agnostic recipes to select the

<sup>\*</sup>Equal contribution <sup>1</sup>The University of Texas at Austin <sup>2</sup>Apple <sup>3</sup>University of North Carolina at Chapel Hill. Correspondence to: Ajay Jaiswal <ajayjaiswal@utexas.edu>.

best recipe is still missing. At this point, one key question arises: *What is the best recipe to identify less knowledgeable experts that can be dropped without sacrificing the vital knowledge and capabilities of the SMoE?*

In this work, we present MoE Experts Compression Suite (MC-Suite), a comprehensive collection of potential recipes for *expert importance estimation* which studies "clues" from four broad and diverse perspectives: a expert & router weight dynamics, b expert inference behavior dynamics, c intermediate activation properties, and d expert gradient properties. In addition to expert importance, MC-Suite unveils numerous valuable insights across experts: dominant experts tend to have lower stable-rank (*i.e.,* pretraining knowledge is well compressed [\(Jaiswal et al.,](#page-10-3) [2024\)](#page-10-3)) which is favorable for additional compression using lowrank factorization; intermediate activation and gradients corresponding to dominant experts tend to have higher entropy indicating better information quantity and conducive finetuning abilities for downstream adaptation [\(Zhang et al.,](#page-12-3) [2024;](#page-12-3) [Zhao et al.,](#page-12-4) [2024\)](#page-12-4); among many others as outlined in Section [3.](#page-4-0) It is important to note that dropping experts involves deleting its entry in the router gating function, which leaves the MoE subnetwork in a sub-optimal state (*i.e.,* increased skewness in load distribution across retained experts, abrupt drop in performance with high dropping ratio). Most existing prior works [\(Lu et al.,](#page-11-7) [2024;](#page-11-7) [He et al.,](#page-10-4) [2024;](#page-10-4) [Muzio](#page-12-2) [et al.,](#page-12-2) [2024\)](#page-12-2) adopt *one-shot* criterion estimation for expert removal that alleviates impact incurred due to sparsification in the form of load imbalance and abrupt performance drop.

In this work, we systematically illustrate that extending oneshot pruning to iterative pruning with re-estimation of importance criterion in k-rounds[1](#page-1-0) , leads to identifying a better subset of experts for dropping. Moreover, motivated by lottery ticket hypothesis [\(Frankle & Carbin,](#page-10-5) [2018\)](#page-10-5), we propose MoE Lottery Subnetwork which involves *task-agnostic budget finetuning*[2](#page-1-1) using next-token prediction objective to address the intermediate sub-optimal state induced due to expert-level sparsification. More specifically, the MoE lottery subnetwork is derived using an iterative *estimationprune-finetune* procedure, and our experiments illustrate that the task-agnostic finetune submodule can help in load distribution across remaining experts along with improving the performance.

To unveil the true merits of expert-level sparsification, in this

work we ask an interesting question: *Given the existence of redundancy across experts, during expert-level sparsification, what capabilities of full-MoE are severely impacted?* We hypothesize that during expert-level sparsification of well-trained MoEs, *instruction following capabilities* are notably hurt while the derived MoE subnetwork still retains the pretraining knowledge and reasoning abilities to a great extent. Our work design controlled experiments from zero-shot setting to k-shot setting and supervised finetuning (SFT) using instruction-tuning dataset, to augment instruction following capabilities into derived MoE subnetwork. Our experimental results indicate that external instructionfollowing support can impressively minimize the performance drop due to expert-level sparsification on complex reasoning downstream tasks. Our key contributions can be briefly summarized as:

- We present MoE Experts Compression Suite (MC-Suite), to re-look the expert importance estimation and facilitate a comprehensive benchmark from a multidimensional perspective. Our extensive experiments show that activation & gradient-guided importance estimation criterions that take into account both input tokens and weight parameters, identifies a superior subset of least dominant experts which can be dropped with minimal impact.
- We explore the potential of iterative estimate-prunefinetune procedure in context of expert-level sparsification. Our experiments illustrate that a fairly limited amount of task-agnostic finetuning facilitate not only improved performance of resultant subnetwork but overcome the skewness in load distribution incurred due expert dropping.
- Our extensive experiments across multiple downstream dataset (*e.g.,* MMLU, ARC-c, ARC-e, HellaSwag, and WinoGrande) surprisingly found that MoE subnetworks, even at non-trivial sparsity ratios (*e.g.*, ≥50% with ≥1.27× speedup and ≤0.55× memory usage) can achieve robust performance subjected to external augmentation of instruction following capabilities using k-shot examples or supervised finetuning.

# <span id="page-1-2"></span>2. MoE Experts Compression Suite (MC-SUITE): An Exhaustive Basket of Strategies to Find Fantastic Experts

Mixture-of-Experts (MoE) architecture has been recently gaining enormous attention for the scaling up of LLMs while maintaining roughly constant FLOPs. By incorporating multiple expert networks and employing a sparse gating mechanism, MoE achieves efficient computation, enabling the development of larger models within the constraints of limited computational resources [\(Fedus et al.,](#page-10-6) [2022;](#page-10-6) [Jiang](#page-11-8)

<span id="page-1-0"></span><sup>1</sup>Our experiments in Section [3.3](#page-7-0) illustrate that subnetworks identified from one-shot vs. iterative pruning are significantly different. We conclude that iterative pruning helps in improving subnetwork quality while additional finetuning helps in retaining the capabilities of subnetwork to avoid abrupt performance degradation.

<span id="page-1-1"></span><sup>2</sup>Our experiments in Appendix [B](#page-13-0) confirms that a limited number of training iterations are sufficient to address the sub-optimal state of MoE subnetwork produced after expert-level sparsification.

![](_page_2_Figure_1.jpeg)

Figure 1. MoE Experts Compression Suite (MC-Suite): A comprehensive basket of criterions (c) to investigate dominant experts across different SMoE blocks from weight, expert behavior, intermediate activations, and gradient behavior perspective. Criterion with  $\odot$  indicate it has been previously explored either in exactly the same formulation or with slight variation. Based on the score of a criterion  $(\text{score}_c^e)$  estimated within a MoE layer, an expert (e) is identified and removed.

![](_page_2_Figure_3.jpeg)

Figure 2. Wikitext Perplexity of Mixtral  $8 \times 7B$  pretrained checkpoint when removing a single expert e from layer l.

et al., 2024). Despite its advantages, MoE suffers from extensive memory costs, which hinder its practical deployment and widespread adoption. For example, the Mixtral-8×7B MoE model takes around 180GB memory while only 28GB parameters are activated for each input token<sup>3</sup>. In parallel to conventional model compression techniques like weight sparsity, quantization, and distillation; the architecture design of MoEs facilitates a unique opportunity for *expert-level sparsification* which involves identifying and removing the least important experts or connections.

Figure 2 presents the wikitext perplexity of Mixtral-8×7B by dropping a single expert e from layer l. It can be clearly **noted** that some experts tend to have an abrupt impact on the performance of the pre-trained checkpoint compared

to others<sup>4</sup>. Therefore, it is critically important to carefully identify the subset of *least important* experts, which are pruned to match the desired sparsity level with minimal impact on performance. In this section, we present MoE Experts Compression Suite (MC-Suite), a first comprehensive benchmark to investigate expert importance using a wide spectrum of novel and previously explored (*e.g.*, expert usage frequency) criterions broadly categorized in four groups: weight-guided expert importance, inference behavior based importance, activation-guided importance, and gradient-guided importance.

### <span id="page-2-0"></span>2.1. Preliminaries and Notations

Consider an MoE-based transformer model  $M_L$  with L MoE layers for processing a set of input tokens  $\mathcal{X} = \{x_1, x_2, ..., x_t\}$ . A standard MoE layer  $(M_l)$  is composed of a set of n experts  $\mathcal{E} = \{E_1, E_2, ..., E_n\}$  with corresponding weights  $\{W_1, W_2, ..., W_n\}$  and a gating function G with weight matrix  $W_G^{d \times n}$ . The gating function is responsible for selecting which experts will be activated for a given input token  $x_i$  by estimating selection score  $G(x_i) \in \mathbb{N}$  with respect to all experts in  $\mathcal{E}$ . The input token  $x_i$  is processed by top-k experts with scaled highest score, and the expert's outputs (intermediate activations)  $\mathcal{A} = \{a_1, a_2, ..., a_k\}$  are combined into a weighted sum based on affinity score provided by the gating function. It

<span id="page-2-1"></span><sup>&</sup>lt;sup>3</sup>The estimates are calculated using full precision (float32).

<span id="page-2-2"></span><sup>&</sup>lt;sup>4</sup>Some Experts are Special: Across our experiments, we found that dropping of special experts lead to abrupt performance drop and this behavior is consistent for different tasks and datasets.

can be summarized as follows:

$$\mathcal{K}_i = \text{top-}k(\text{softmax}(\boldsymbol{G}(x_i)), k) \tag{1}$$

$$y_i = \sum_{m \in \mathcal{K}_i} \mathbf{G}_m(x_i) \cdot \mathbf{E}_m^{\mathbf{W}_m}(x_i)$$
 (2)

where  $\mathcal{K}_i$  indicated the top-k indices of the selected experts for token  $x_i$ ,  $G_m(x_i)$  and  $E_m^{W_m}$  represents the affinity score and output for m-th expert for token  $x_i$ . min or max corresponds to minimum and maximum value of the criterion across all experts.

### 2.2. Weight-Guided Expert Importance

① Expert Weight Similarity Criterion (EWS): In this criterion, we flatten the weights of all experts of layer l of M and calculate pairwise cosine similarity across them. Depending on the min or max argument, we select expert  $E_p$  which have min or max cosine similarity with  $\mathcal{E} - \{E_p\}$ .

$$\begin{aligned} \cos_{n\times n} &= \texttt{pairwise-cos}_{\forall (p,q)\in\mathcal{E}\times\mathcal{E}}(\texttt{flatten}(\boldsymbol{W}_{\boldsymbol{E}_p})) \\ &\text{drop-index} &= \texttt{min/max}_{\forall p\in\mathcal{E}}\big\{\texttt{sum}(\cos_{[p,:]}) - \cos_{[p,p]}\big\} \end{aligned} \tag{3}$$

② Router Weight Norm Criterion (RWN): Given a token, the router gating function is responsible for selecting top-k experts from n available experts using its weight matrix  $W_G^{d\times n}$ . RWN aims to understand the role of the gating weights corresponding to  $E_p$  in  $W_G$  to estimate its importance

$$\label{eq:drop-index} \begin{aligned} \text{drop-index} &= \min / \max \big\{ \text{norm}_{l2}(\boldsymbol{W}_{\boldsymbol{G}}^{d \times n}, \text{dim=1}) \big\} \end{aligned} \tag{4}$$

③ Expert Weight Stable Rank Criterion (WSR): Stable rank of an expert weight matrix ( $W_{E_p}$ ) is defined as  $\frac{\sum_{i=1}^r \sigma_i^2(W_{E_p})}{\sigma_1^2(W_{E_p})}$ , where  $\sigma_i$  refers to the *i*-th sorted singular value of  $W_{E_p}$ . Recently, stable-rank has been studied in the context of LLM layer importance, generalizability, and downstream adaption ability (Sanyal et al., 2020; Jaiswal et al., 2024; Zhang et al., 2024) and we aim to extend it for estimation of expert importance.

$$\operatorname{drop-index} = \min / \max \big\{ \operatorname{stable-rank}_{\forall p \in \mathcal{E}}(\boldsymbol{W}_{\boldsymbol{E}_p}) \big\} \tag{5}$$

**4** Expert Weight Norm Criterion (EWN): In this criterion, we calculate the l2-norm of weights of all experts of layer l of model M. Depending on the min or max argument, we select expert  $E_p$  that has min or max weight norm for dropping.

$$drop-index = min/max\{norm_{l2}(\boldsymbol{W}_{\boldsymbol{E}_n})\}$$
 (6)

#### 2.3. Inference-Guided Expert Importance

① Expert Usage Frequency Criterion (EUF): In this criterion, we define expert usage with a calibration dataset (e.g., C4 validation for MC-Suite). Expert usage is estimated by the ratio of tokens that activate  $E_p$  with a fixed calibration set. Note that we experimentally found that expert usage frequency is **not** strongly tied to the choice of calibration dataset. Given  $\mathcal{X}$  as calibration set with t-tokens and  $\mathcal{K}_i$  be the top-k experts for token i, we select expert  $E_p$  as:

$$\operatorname{drop-index} = \min / \max_{\forall p \in \mathcal{E}} \left\{ \sum_{x_i \in \mathcal{X}} \mathbb{1}[\mathcal{K}_i \cap \{E_p\}] \neq \emptyset \right\}$$
 (7)

② Expert-Expert Collaboration Criterion (ECC): Expert-Expert collaboration count is as defined as the number of times two experts  $E_p$  and  $E_q$  are selected to process a token  $x_i$ . Let  $\mathcal{X}$  as calibration set with t-tokens and  $\mathcal{K}_i$  be the top-k experts for token i, we define:

collaboration-matrix
$$_{(\mathbf{E}_{p},\mathbf{E}_{q})\in(\mathcal{E}\times\mathcal{E})}^{n\times n} = \sum_{x_{i}\in\mathcal{X}}\mathbb{1}[\mathcal{K}_{i}\cap\{E_{p},E_{q}\} == \{E_{p},E_{q}\}]$$
(8)

Given the collaboration matrix, we select the expert pair  $(E_p, E_q)$  wrt. the min or max argument and drop-index is identified as the expert that tends to have lower usage frequency.

③ Expert Vocabulary Coverage Criterion (EVTC): Expert vocabulary coverage is defined as the fraction of unique tokens from the model vocabulary, which is processed by a given expert  $\mathcal{E}_p$ . Consider  $\mathcal{V}$  be the model vocabulary and  $\mathcal{X}_p$  are the tokens from calibration set  $\mathcal{X}$  which are routed to expert  $\mathbf{E}_p$  by gating function, we select  $\mathbf{E}_p$  as:

$$\operatorname{drop-index} = \min / \max_{\boldsymbol{y} \in \mathcal{E}} \left\{ \operatorname{unique}(\mathcal{X}_p) / |\mathcal{V}| \right\} \quad (9)$$

**4 Expert Input Token Similarity (ETS):** In this criterion, we aim to estimate the input token-level similarity across experts. More specifically, with  $\mathcal{X}_p$  as the tokens routed to expert  $\boldsymbol{E}_p$ , we generate:

$$\begin{array}{l} \operatorname{token-similarity-matrix}_{(\boldsymbol{E}_p,\boldsymbol{E}_q)\in(\mathcal{E}\times\mathcal{E})}^{n\times n} = \operatorname{count}(\mathcal{X}_p\cap\mathcal{X}_q) \\ \operatorname{Given the token similarity matrix, we select the expert pair} \end{array}$$

Given the token similarity matrix, we select the expert pair  $(E_p, E_q)$  wrt. the min or max argument and drop-index is identified as the expert that tends to have lower usage frequency.

#### 2.4. Activation-Guided Expert Importance

① Expert Activation Similarity Criterion (EAS): Given the calibration set of tokens  $\mathcal{X}$ , we accumulate the activation of tokens routed to experts  $(\mathcal{A}_{E_p})$  using forward hooks. We first generate the activation similarity matrix across each

expert pair depending on min or max argument, we select expert  $E_p$  which have min or max activation similarity with  $\mathcal{E}-\{E_p\}$ .

$$\begin{aligned} \text{activation-similarity}_{(E_p, E_q)}^{n \times n} &= \frac{1}{|\mathcal{A}_{E_p}| \times |\mathcal{A}_{E_q}|} \\ &\sum_{(a_m, a_n) \in (\mathcal{A}_{E_p} \times \mathcal{A}_{E_q})} \text{cosine}(a_m, a_n) \end{aligned}$$

 $\mathsf{drop\text{-}index} = \texttt{min/max}_{\forall p \in \mathcal{E}} \big\{ \texttt{sum}(\mathsf{activation\text{-}similarity}_{[p,:]})$ 

$$-\text{activation-similarity}_{[p,p]}$$
 (11)

② Expert Activation Entropy Criterion (EAE): Entropy is the measurement of information quantity and we extended (Lin et al., 2024) entropy quantification strategy for convolution feature maps to expert activation. More specifically, in MC-Suite, the entropy of an expert activation ( $A_{E_p}$ ) is proportional to the summation of the logarithm of the standard deviation of each hidden dimension:

$$H(\mathcal{A}_{E_p}) \propto \sum_{j} \log[\sigma(\mathcal{A}_{E_p}^{j})]$$
 (12)

where,  $\sigma(\mathcal{A}_{E_p}^j)$  calculate the standard deviation of  $j_{th}$  hidden dimension of the activation and sum it to obtain activation entropy and select expert  $E_p$  which have min or max activation entropy.

- ③ Expert Activation Distribution Outliers (EAO): In this criterion, we estimate outliers in the normally distributed activation of experts. More specifically, given  $\mathcal{A}_{E_p}$  as the activations of expert  $E_p$ , we estimate mean  $(\mu_{\mathcal{A}_{E_p}})$  and standard deviation  $(\sigma_{\mathcal{A}_{E_p}})$  across the hidden dimension and count outliers outside the interval  $\mu_{\mathcal{A}_{E_p}} \pm c \times \sigma_{\mathcal{A}_{E_p}}$  with value of c = 3.
- **(4)** Expert Activation Norm (EAN): In this criterion, we calculate the l2-norm across the hidden dimension for the accumulated activation ( $\mathcal{A}_{E_p}$ ) of expert  $E_p$ . Overall activation norm of  $E_p$  is estimated as the sum of l2-norm over all hidden dimensions and the drop-index is given as:

$$drop-index = \min/\max_{\forall p \in \mathcal{E}} \left\{ sum(norm_{l2}(\mathcal{A}_{E_p}, dim=0)) \right\}$$
(13)

### 2.5. Gradient-Guided Expert Importance

(1) Expert Gradient Similarity Criterion (EAS): Given the calibration set of tokens  $\mathcal{X}$ , we first pass it through the model in batches and accumulate the gradient for all the expert's weight matrices. Consider  $W_{E_p}^g$  be the gradient corresponding to the weight matrix of expert  $E_p$ . We flatten the gradient matrix for all experts of layer l and calculate the pairwise cosine similarity across them.

$$\begin{aligned} \cos_{n\times n} &= \operatorname{pairwise-cos}_{\forall (p,q)\in\mathcal{E}\times\mathcal{E}}(\operatorname{flatten}(\boldsymbol{W}_{\boldsymbol{E}_p}^g)) \\ &\operatorname{drop-index} &= \operatorname{min/max}_{\forall p\in\mathcal{E}} \left\{\operatorname{sum}(\cos_{[p,:]}) - \cos_{[p,p]}\right\} \\ &(14) \end{aligned}$$

(2) Expert Gradient Entropy Criterion (EAE): Gradient entropy is a measurement of information encoded (Guan et al., 2019) within them, and it can be a well-suited indicator for judging the expert importance with the privilege of finetuning. Similar to activation entropy, we estimate gradient entropy by calculating the standard deviation across the hidden dimension of accumulated gradients as:

$$H(\boldsymbol{W}_{\boldsymbol{E}_p}^g) \propto \sum_{i} \log[\sigma(\boldsymbol{W}_{\boldsymbol{E}_p}^{g^j})]$$
 (15)

- ③ Expert Gradient Outliers Criterion (EAO): In this criterion, we estimate the number of outliers in the accumulated gradients of experts. Given  $W_{E_p}^g$  corresponding to weight of expert  $E_p$ , we count number of outliers outside interval  $\mu_{W_{E_p}^g} \pm c \times \sigma_{W_{E_p}^g}$  with value of c = 3.
- **4** Expert Gradient Norm Criterion(EAN): In this criterion, we calculate the l2-norm of gradients of weights for all experts of layer l of model M. Depending on the min or max argument, we select expert  $E_p$  that has min or max gradient norm for dropping.

<span id="page-4-1"></span>
$$\text{drop-index} = \min / \max \left\{ \text{norm}_{l2}(\boldsymbol{W}_{\boldsymbol{E}_n}^g) \right\} \quad \ (16)$$

![](_page_4_Figure_19.jpeg)

Figure 3. Overview of Different Expert Pruning Strategies: Given a target expert sparsity of S%, (a) One-shot pruning: removes S% of experts from each layer L from MoE based on one-time estimation of criterion c; (b) Iterative pruning: removes S/k% of experts before re-estimation of criterion c for k-rounds; (a) MoE Lottery pruning: removes S/k% of experts followed by task-agnostic budget finetuning using calibration data before reestimation of criterion c for k-rounds.

# <span id="page-4-0"></span>3. MoE Lottery Subnetworks: Blessing From task-agnostic budget fintetuning

Expert-level sparsification of SMoEs involves identifying r experts with the least importance using criterions outlined in Section 2 and discarding them to reduce exorbitant memory requirements of loading n experts. Dropping experts require explicit handling of the routing gate function by removing the entry corresponding to dropped experts. In our work, we found that gating function is highly sensitive to any modification and an ad-hoc deletion of r entries from the router matrix (i.e.,  $\mathbf{W}^{d\times n} \to \mathbf{W}^{d\times n-r}$ ) not only lead to significant performance degradation but also induces heavier load on few among remaining n-r experts. Prior

![](_page_5_Figure_1.jpeg)

Figure 4. Performance comparison (perplexity on C4) of Mixtral-8×7B Base Lottery Subnetworks identified by dropping experts iteratively using various criterions from MC-Suite. Original Mixtral-8×7B Base checkpoint achieves 7.44 perplexity on C4 validation set. Min & Max represents an expert (e) with minimum/maximum score of a criterion (c).

works have limited exploration of *one-shot* removal of r experts to achieve a sparsity ratio of s% and overlooked attention at finetuning to address the sub-optimal state of SMoE subnetwork after sparsification.

In this work we adopt motivation from the success of lottery ticket hypothesis (Frankle & Carbin, 2018; 2019) and explore: ① iterative pruning of experts in k-rounds to attain sparsity ratio of s%; ② incorporation of task-agnostic finetuning on next token prediction task to stabilize the suboptimal state of SMoE subnetworks. Moreover, an iterative pruning strategy with re-estimation of importance criterion enables taking into account the impact of thee removal of the first round of experts on deciding the importance of remaining experts. We propose MoE Lottery Subnetwork, which relies on iterative estimate-prune-finetune procedure as shown in Figure 3. Note that we choose to state budget finetuning because we found that one doesn't require extensive finetuning iterations but a marginal amount is sufficient to obtain desirable performance gains (Appendix B).

Our experimental results in this section have two-folds. Firstly, we perform a comprehensive evaluation of the criterions of MC-Suite (Section 2) using MoE lottery subnetworks with varying sparsity ratios of  $s \in \{12.5\%, ..., 75.0\%\}$ . Secondly, we aim to understand the merits of iterative pruning and task-agnostic budget finetuning by selecting top-performing MC-Suite criterions.

### 3.1. MC-Suite and MoE Lottery Subnetworks

MC-Suite consists of a series of criterions from four diverse perspectives that provide "clues" for identifying experts that <span id="page-5-0"></span>contribute least to the original SMoE model and thus can be discarded. Given a criterion c from MC-Suite, we study both maximizing and minimizing c while generating the MoE lottery subnetworks to understand the characteristics of retained experts and its impact on the final performance. Figure 4 presents the C4 validation perplexity of MoE lottery subnetworks of Mixtral-8×7B Base model where an expert e from a MoE layer l is dropped subjected to maximum or minimum value of c across other fellow experts in l. Table 1 presents the comparison of best-performing criterions from four different perspectives of MC-Suite along with randomly selected expert dropping baseline. It can be clearly observed that the usage of criterions from MC-Suite significantly helps in improving the performance of MoE lottery subnetworks. In our experimental setting, we choose to drop 32 experts (i.e., 12.5% sparsity) in every round of iterative pruning with one expert per layer. Our experiments found that a non-uniform dropping of experts per layer by estimating c globally creates bottleneck layers, with some layers having significantly high sparsity while some remain unpruned, leading to diminished finetuning benefits and sharding simplicity.

The benefits of MC-Suite are <u>not</u> limited to exploration of the best recipe to identify least important experts for dropping, but extends in deriving many valuable hidden insights of important experts. We comprehend few interesting findings as: ① activation and gradient-guided criterions (minimum activation norm and gradient entropy) that take into account both input tokens and model parameters achieves the *superior performance* over conventional criterions such as expert usage, expert weight similarity,

| Criterion (c)                     | 12.5% | 25.0% | 37.5% | 50.0% | 62.5% | 75.0% |
|-----------------------------------|-------|-------|-------|-------|-------|-------|
| Random Dropping (One-shot)        | 9.01  | 11.02 | 11.95 | 15.21 | 21.10 | 34.47 |
| Random Dropping (Iterative)       | 9.78  | 11.12 | 13.06 | 15.46 | 22.76 | 38.94 |
| Random Dropping (w. MoE Lottery)  | 9.66  | 10.54 | 11.83 | 13.71 | 18.23 | 33.05 |
| Max-Router Weight Norm (RWN)      | 8.47  | 9.00  | 9.87  | 10.70 | 13.50 | 17.26 |
| Max-Expert Token Similarity (ETS) | 8.28  | 8.82  | 9.50  | 10.43 | 12.48 | 16.03 |
| Min-Expert Gradient Entropy (EGE) | 8.17  | 8.84  | 9.54  | 10.45 | 11.88 | 15.08 |
| Min-Expert Activation Norm (EAN)  | 8.18  | 8.63  | 9.21  | 9.99  | 11.43 | 14.02 |

Table 1. Performance comparison (perplexity on C4) of Mixtral-8×7B Instruct Lottery Subnetworks identified by various top-performing criterions from MC-Suite. Original Mixtral-8×7B Instruct checkpoint achieves 7.82 perplexity on C4 validation set.

| % Experts Dropped    | Random Dropping |           | Min-Activation Norm (Min-EAN) |          |           | Min-Gradient Entropy (Min-EGE) |          |           |             |
|----------------------|-----------------|-----------|-------------------------------|----------|-----------|--------------------------------|----------|-----------|-------------|
| ie Ziiperio Ziropped | One-shot        | Iterative | MoE Lottery                   | One-shot | Iterative | MoE Lottery                    | One-shot | Iterative | MoE Lottery |
| 0%                   |                 |           |                               |          | 7.44      |                                |          |           |             |
| 12.5%                | 11.25           | 7.94      | 7.89                          | 7.95     | 7.90      | 7.89                           | 7.89     | 7.89      | 7.88        |
| 25.0%                | 12.74           | 10.98     | 11.01                         | 8.56     | 8.53      | 8.38                           | 8.47     | 8.41      | 8.26        |
| 37.5%                | 13.89           | 13.19     | 12.22                         | 12.87    | 9.35      | 9.00                           | 13.33    | 9.48      | 9.09        |
| 50.0%                | 17.08           | 15.85     | 14.13                         | 14.74    | 10.44     | 9.76                           | 15.37    | 10.72     | 9.88        |
| 62.5%                | 30.41           | 18.79     | 20.60                         | 21.36    | 12.55     | 11.00                          | 22.21    | 12.81     | 11.00       |
| 75.0%                | 36.92           | 32.73     | 27.33                         | 30.59    | 17.39     | 13.05                          | 35.83    | 17.70     | 13.09       |

Table 2. Improved Language Modeling Abilities: Performance comparison of MoE Lottery Subnetworks identified using criterion (c) with respect to Iterative and One-shot pruning. MoE Lottery Subnetworks, which are supplemented with task-agnostic finetuning, are able to restore a better optimal state impacted by ad-hoc derivation from their dense counterpart.

etc.; 2 surprisingly, l2-norm of router weight matrix turn out to be the best performing candidate in comparison to other expert weight based criterions; (3) dropping experts with higher vocabulary coverage lead to a significant drop in performance which indicate efforts to improve specialization across experts in MoEs can be non-conducive for expert-level sparsification; (4) dominant experts tends to have lower stable-rank, which aligns with recent findings of (Jaiswal et al., 2024; Zhang et al., 2024) that LLMs weight matrices which are critical and well-trained also have comparatively lower stable-rank with further compression potential with orthogonal techniques like low-rank factorization; (5) our **novel** criterion of *entropy quantification of activa*tion and gradient aiming to measures information encoded within them, turns out to best performing recipes for estimating expert importance and also favorable for downstream task finetuning. Interestingly, while comparing the impact of expert-level sparsification for Mixtral-8×7B Base and Instruct checkpoints, we found that task-agnostic finetuning has comparatively lower benefits for Instruct in comparison to Base model suggesting to perform expert dropping before instruction tuning.

# 3.2. Understanding the Merits of Task-Agnostic Budget Finetuning

In this section, we attempt to unveil the true merits of the iterative *estimate-prune-finetune* procedure of MoE lottery subnetworks. To investigate the benefits contributed by iterative pruning and task-agnostic finetuning, we present perfor-

<span id="page-6-1"></span><span id="page-6-0"></span>mance comparison for one-shot, iterative pruning, and MoE lottery subnetworks. Firstly, Table 2 illustrate the improved language modeling abilities measured using validation perplexity of C4 dataset where MoE lottery networks (with Min-EAN and Min-EGE criterion) can achieve  $\sim 3 \times$  better performance compared to one-shot pruning, while iterative pruning without any finetuning can still achieve  $\sim 2 \times$  superior performance. It is also interesting to note that even the random expert selection baseline significantly benefits from iterative pruning and finetuning with  $\sim 9.5$  points better perplexity than one-shot pruning. Secondly, Table 3 presents the improved zero-shot downstream performance (no in-context examples) of MoE lottery subnetworks over one-shot and iterative pruning at varying sparsity levels on MMLU and WinoGrande. Clearly, it can be observed that while one-shot pruning starts performing worse than random guess with merely a 25% sparsity ratio; MoE lottery networks performance doesn't drop below random guess even at non-trivial sparsity ratio (62.5%-75.0%). Moreover, the the contribution of iterative estimate-prune-finetune become more notable with increasing sparsity ratios.

Next, we ask an interesting question: How does task-agnostic finetuning, which aims to re-adjust the router weight, influence the load distribution across experts? To this end, Figure 6 illustrates the expert load distribution<sup>5</sup> of remaining experts of a MoE layer from Mixtral-8×7B

<span id="page-6-2"></span> $<sup>^5</sup>$ Expert (e) Load: Given a fixed number of input tokens, # tokens processed by the expert e.

|            | Criterion (c)= Min-Activation Norm | 0%    | 12.5% | 25%   | 37.5% | 50%   | 62.5% | 75%   |
|------------|------------------------------------|-------|-------|-------|-------|-------|-------|-------|
| MMLU       | One-shot Pruning                   |       | 52.97 | 43.97 | 13.55 | 18.91 | 12.63 | 5.82  |
|            | Iterative Pruning                  | 60.01 | 48.51 | 47.81 | 45.63 | 3574  | 29.71 | 23.88 |
|            | MoE Lottery Networks               |       | 49.54 | 49.65 | 47.13 | 40.79 | 37.24 | 28.12 |
| WinoGrande | One-shot Pruning                   |       | 55.13 | 50.09 | 37.45 | 36.91 | 20.44 | 24.63 |
|            | Iterative Pruning                  | 56.59 | 55.90 | 52.17 | 49.96 | 48.53 | 47.11 | 50.35 |
|            | MoE Lottery Networks               |       | 55.92 | 52.98 | 50.96 | 49.56 | 49.30 | 50.74 |

Table 3. Improved Zero-shot Downstream Performance: Downstream task performance comparison of MoE Lottery Subnetworks identified using criterion (c) with respect to Iterative and One-shot pruning in zero-shot setting (no in-context examples). MoE Lottery networks tend to have superior abilities to follow instructions required to complete the downstream tasks.

![](_page_7_Figure_3.jpeg)

Figure 5. Dropped Experts Distribution with 50% Sparsity: (a) Difference of experts identified to be dropped with *one-shot pruning* in comparison with *moe-lottery pruning*, (b) Difference of experts identified to be dropped with *iterative pruning* in comparison with *moe-lottery pruning*. Light Bisque color corresponding to an expert  $(e_L^i)$  indicates agreement across both pruning techniques to drop  $e_L^i$ , Dark pink indicates disagreement to drop, while Black indicates agreement to retain  $e_L^i$ .

![](_page_7_Figure_5.jpeg)

Figure 6. Improved load balancing across experts (l=6~&~30) for Mixtral-8×7B Base model before and after task-agnostic finetuning with C4.

Base model with 50% expert sparsity ratio before (dashed red line) and after (solid green line) task-agnostic finetuning using C4 dataset. It can be clearly observed that our proposed finetuning subroutine can significantly help in induced skewness in load distribution across experts due to expert droping and removal of its entry from the router gating function. Note that a well-balanced load distribution across experts is encouraged to facilitate better GPU memory utilization and speedup.

# <span id="page-7-0"></span>3.3. Understanding Expert dropping pattern Across One-shot, iterative & MoE Lottery Pruning

In this section, we study the divergence of the selection of experts for pruning of *one-shot* and *iterative pruning* w.r.t. *MoE lottery pruning*. The primary aim of this study is to highlight the benefits of iterative pruning with re-estimation of expert importance criterions. It can be clearly observed from Figure 5(a) that there exists a *significantly high disagreement* (dark pink) between one-shot and iterative pruning while selecting least dominant experts *leading to completely different resultant subnetworks*. The substandard performance of the one-shot method indicates that the iden-

<span id="page-7-3"></span><span id="page-7-2"></span><span id="page-7-1"></span>tified subnetwork is not of high quality in comparison to iterative pruning. On the other hand, Figure 5(b) illustrates a notable high agreement across experts, which undergoes dropping to achieve a sparsity ratio of 50%. This leads to an interesting conclusion that task-agnostic finetuning does not significantly alter the expert selection choice selection but instead helps in addressing the impact incurred due to sparsification in the form of load imbalance and abrupt performance drop.

# 4. What is Lost v/s what prevails? An in-depth Investigation of Expert Dropping and Lost capabilities

SMoE models require enormous memory to host experts during inference while being known to have poor utilization of its capacity. In recent times, multiple LLM compression techniques (e.g., weight sparsity, quantization, low-rank factorization, etc.) are being developed to address the memory and computational bottleneck. Some works (Jaiswal et al., 2023a; Hong et al., 2024; Yin et al., 2023a) attempt to understand the impact of compression on pretrained checkpoints while handling knowledge-intensive tasks, trust, and safety. Motivated by their findings, we aim to understand the impact of dropping least important and redundant experts during expert-level sparsification of SMoEs. Given that SMoEs are trained using a Top-k routing policy, each token is processed by k experts, promoting redundancy and less sensitivity to expert dropping by design choice. We ask: What capabilities of full-SMoEs are severely impacted by the removal of least dominant experts?

![](_page_8_Figure_1.jpeg)

Figure 7. Downstream task performance of MoE Lottery Subnetworks at varying sparsity level when augmented with external instruction following capabilities using k-shot examples (Row 2) and supervised finetuning (Row 3) using instruction-tuning dataset.

At first, a *narrow view* of the zero-shot downstream evaluation of SMoE subnetworks with expert-level sparsification indicates a sharp performance drop compared to the full-SMoEs. Figure 7 (row 1) illustrates the zero-shot performance of MoE lottery subnetworks identified with four criterions from MC-Suite on 5 popular reasoning and knowledge-intensive tasks. It can be clearly observed that the expert-dropping tends to have an acute impact on the downstream tasks <u>but</u> we pause and ask: *Is this abrupt performance degradation incurring due to loss of pretraining knowledge and reasoning abilities or instruction-following abilities?* We **conjecture** that when we drop the least dominant experts, SMoEs instruction following capabilities are predominantly hurt, and it can be restored to a notable extent with external support.

To experimentally validate our conjecture, we design the controlled experiments in three folds: ① zero-shot setting which directly evaluate pruned SMoE performance on downstream tasks without any in-context example; ② k-shot setting which provide k in-context examples as external assistance for compressed LLMs to follow downstream instructions; ③ supervised finetuning (SFT) that aim to explicitly embed external instruction following support in compressed SMoE checkpoint by finetuning using instruction following dataset. Figure 7 (row 2 & 3) illustrates that external instruction-following support can impressively minimize the performance gap due to expert-level sparsification on complex reasoning downstream tasks. Note that for fair comparison, our full-SMoE baselines represented as straight lines are also provided exactly similar external instruction-

<span id="page-8-0"></span>following support. Interestingly, we can observe that SFT, even with the zero-shot setting, can enable **robust** performance of compressed SMoE models at non-trivial sparsity ratios ( $\geq 50\%$ ). Moreover, for some comparatively easier tasks (*e.g.*, BoolQ, ARC-easy), it facilitates pruned SMoEs to outperform the full-SMoE baseline.

# 5. Expert Dropping v/s LLM Weight Pruning Techniques

LLM weight pruning algorithm (Yin et al., 2023b; Jaiswal et al., 2023b; Sun et al., 2023; Frantar & Alistarh, 2023) involves removing non-significant weights parameters by setting them to zero. Recent hardware advancements have enabled practical speedup for structural N:M sparsity patterns (Nvidia, 2020; Zhou et al., 2021). In this section, we investigate the downstream task performance of the expertlevel sparsification method with the representative weight pruning baselines (random, magnitude, and wanda). For expert-level sparsification, we present MoE lottery networks with random and minimum activation norm criterions to identify dominant experts. Provided the hardware supported 2:4 weight sparsity patterns, we choose expert drop ratio (r=4) per layer to achieve 50% sparsification for both categories for fair comparison.

Table 4 summarizes the performance comparison in zero-shot setting for all baselines and MoE Lottery subnetwork for Mixtral-8×7B Base and Instruct checkpoints. It can be observed that expert-level sparsification can achieve  $\sim 3.6\%$  average performance gain over the Wanda pruning

| Model        | Method                 | Sparsity | Arc-c | ARC-e | HellaSwag | MMLU  | WinoGrande | Average |
|--------------|------------------------|----------|-------|-------|-----------|-------|------------|---------|
|              | None                   | r = 8    | 78.18 | 91.94 | 64.88     | 60.01 | 56.59      | 70.32   |
|              | Random Pruning         | 2:4      | 19.47 | 48.90 | 28.90     | 17.05 | 22.07      | 27.27   |
| Mixtral 8×7B | Magnitude Pruning      | 2:4      | 31.07 | 69.76 | 43.23     | 42.77 | 38.56      | 45.07   |
| MIXTRAI 8×7B | Wanda Pruning          | 2:4      | 43.82 | 70.16 | 53.16     | 50.21 | 48.96      | 52.91   |
|              | Min-EAN Expert Pruning | r = 4    | 60.02 | 71.41 | 50.78     | 51.33 | 49.56      | 56.62   |
|              | None                   | r = 8    | 81.86 | 93.21 | 78.06     | 64.67 | 63.77      | 76.31   |
|              | Random Pruning         | 2:4      | 23.68 | 56.42 | 37.01     | 22.15 | 29.07      | 31.94   |
| Mixtral 8×7B | Magnitude Pruning      | 2:4      | 54.96 | 69.44 | 57.18     | 29.08 | 40.79      | 50.29   |
| Instruct     | Wanda Pruning          | 2:4      | 61.92 | 80.23 | 62.90     | 51.05 | 55.30      | 62.28   |
|              | Min-EAN Expert Pruning | r = 4    | 68.50 | 83.59 | 64.46     | 48.56 | 54.65      | 63.95   |

Table 4. Expert-level Sparsification V/s LLM Weight Pruning: Downstream task performance comparison in zero-shot setting (no in-context example) of Mixtral 8×7B base and Instruct when compressed using expert-level sparsification techniques v/s SoTA LLM pruning methods.

while a notable  $\sim 16.2\%$  improvement on ARC-c downstream task. In addition, we also find that the performance benefits for Base model is comparatively superior than Instruct suggesting it is favorable to perform expertlevel dropping on the Base model before instruction tuning.

#### 6. Conclusion

In this paper, we provide a detailed investigation of multiple expert importance estimation techniques (MC-Suite) to identify the best recipe for selecting the least knowledgeable experts that can be dropped without sacrificing the vital knowledge and capabilities of the SMoE. We propose to adopt a iterative pruning strategy with task-agnostic finetuning as a correction measure to minimize the drastic impact on SMoE capabilities. We present and experimentally validate an interesting conjecture that during expert dropping, SMoE instruction following capabilities are predominantly hurt, and SMoE performance can be notably recovered with a few-shot demonstration or supervised finetuning. In our future work, we plan to investigate and disentangle the instruction-following abilities and pretraining knowledge across the parameters of SMoE experts.

#### References

<span id="page-9-7"></span>Artetxe, M., Bhosale, S., Goyal, N., Mihaylov, T., Ott, M., Shleifer, S., Lin, X. V., Du, J., Iyer, S., Pasunuru, R., Anantharaman, G., Li, X., Chen, S., Akin, H., Baines, M., Martin, L., Zhou, X., Koura, P. S., O'Horo, B., Wang, J., Zettlemoyer, L., Diab, M., Kozareva, Z., and Stoyanov, V. Efficient large scale language modeling with mixtures of experts, 2022. URL https://arxiv.org/abs/2112.10684.

<span id="page-9-5"></span>Bandari, A., Yin, L., Hsieh, C.-Y., Jaiswal, A., Chen, T., Shen, L., Krishna, R., and Liu, S. Is c4 dataset optimal for pruning? an investigation of calibration data for llm pruning. In *Proceedings of the 2024 Conference on* 

<span id="page-9-1"></span>Empirical Methods in Natural Language Processing, pp. 18089–18099, 2024.

<span id="page-9-2"></span>Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.14165.

<span id="page-9-6"></span>Chen, T., Huang, S., Xie, Y., Jiao, B., Jiang, D., Zhou, H., Li, J., and Wei, F. Task-specific expert pruning for sparse mixture-of-experts, 2022. URL https://arxiv.org/abs/2206.00277.

<span id="page-9-0"></span>Chen, T., Zhang, Z., Jaiswal, A., Liu, S., and Wang, Z. Sparse moe as the new dropout: Scaling dense and self-slimmable transformers. *arXiv preprint arXiv:2303.01610*, 2023.

<span id="page-9-3"></span>Dai, D., Deng, C., Zhao, C., Xu, R. X., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., Xie, Z., Li, Y. K., Huang, P., Luo, F., Ruan, C., Sui, Z., and Liang, W. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2401.06066.

<span id="page-9-4"></span>DeepSeek-AI, Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Xu, H., Yang, H., Zhang, H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L., Liang, J., Guo, J., Ni, J., Li, J., Chen, J., Yuan, J., Qiu, J., Song, J., Dong, K., Gao, K., Guan, K., Wang, L., Zhang, L., Xu, L., Xia, L., Zhao, L., Zhang, L., Li, M., Wang, M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N., Huang, P., Wang, P., Zhang, P., Zhu, Q., Chen, Q., Du, Q., Chen, R. J., Jin, R. L., Ge, R., Pan, R., Xu, R., Chen,

- R., Li, S. S., Lu, S., Zhou, S., Chen, S., Wu, S., Ye, S., Ma, S., Wang, S., Zhou, S., Yu, S., Zhou, S., Zheng, S., Wang, T., Pei, T., Yuan, T., Sun, T., Xiao, W. L., Zeng, W., An, W., Liu, W., Liang, W., Gao, W., Zhang, W., Li, X. Q., Jin, X., Wang, X., Bi, X., Liu, X., Wang, X., Shen, X., Chen, X., Chen, X., Nie, X., Sun, X., Wang, X., Liu, X., Xie, X., Yu, X., Song, X., Zhou, X., Yang, X., Lu, X., Su, X., Wu, Y., Li, Y. K., Wei, Y. X., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zhao, Y., Sun, Y., Li, Y., Wang, Y., Zheng, Y., Zhang, Y., Xiong, Y., Zhao, Y., He, Y., Tang, Y., Piao, Y., Dong, Y., Tan, Y., Liu, Y., Wang, Y., Guo, Y., Zhu, Y., Wang, Y., Zou, Y., Zha, Y., Ma, Y., Yan, Y., You, Y., Liu, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Huang, Z., Zhang, Z., Xie, Z., Hao, Z., Shao, Z., Wen, Z., Xu, Z., Zhang, Z., Li, Z., Wang, Z., Gu, Z., Li, Z., and Xie, Z. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. URL <https://arxiv.org/abs/2405.04434>.
- <span id="page-10-1"></span>Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. Qlora: Efficient finetuning of quantized llms. *ArXiv*, abs/2305.14314, 2023. URL [https:](https://api.semanticscholar.org/CorpusID:258841328) [//api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:258841328) [258841328](https://api.semanticscholar.org/CorpusID:258841328).
- <span id="page-10-12"></span>Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. Glam: Efficient scaling of language models with mixture-ofexperts. In *International Conference on Machine Learning*, pp. 5547–5569. PMLR, 2022.
- <span id="page-10-16"></span>Dubey, A., Chatterjee, M., and Ahuja, N. Coreset-based neural network compression, 2018. URL [https://](https://arxiv.org/abs/1807.09810) [arxiv.org/abs/1807.09810](https://arxiv.org/abs/1807.09810).
- <span id="page-10-15"></span>Fang, G., Ma, X., Song, M., Mi, M. B., and Wang, X. Depgraph: Towards any structural pruning, 2023. URL <https://arxiv.org/abs/2301.12900>.
- <span id="page-10-6"></span>Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-10-5"></span>Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. *arXiv preprint arXiv:1803.03635*, 2018.
- <span id="page-10-0"></span>Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In *International Conference on Learning Representations*, 2019. URL [https://openreview.net/forum?](https://openreview.net/forum?id=rJl-b3RcF7) [id=rJl-b3RcF7](https://openreview.net/forum?id=rJl-b3RcF7).
- <span id="page-10-11"></span>Frantar, E. and Alistarh, D. Sparsegpt: Massive language models can be accurately pruned in one-shot. In *International Conference on Machine Learning*, pp. 10323– 10337. PMLR, 2023.

- <span id="page-10-2"></span>Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq: Accurate post-training quantization for generative pre-trained transformers. *ArXiv*, abs/2210.17323, 2022. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:253237200) [org/CorpusID:253237200](https://api.semanticscholar.org/CorpusID:253237200).
- <span id="page-10-14"></span>Gou, J., Yu, B., Maybank, S. J., and Tao, D. Knowledge distillation: A survey. *International Journal of Computer Vision*, 129(6):1789–1819, March 2021. ISSN 1573-1405. doi: 10.1007/s11263-021-01453-z. URL [http://dx.](http://dx.doi.org/10.1007/s11263-021-01453-z) [doi.org/10.1007/s11263-021-01453-z](http://dx.doi.org/10.1007/s11263-021-01453-z).
- <span id="page-10-7"></span>Guan, C., Wang, X., Zhang, Q., Chen, R., He, D., and Xie, X. Towards a deep and unified understanding of deep neural models in nlp. In *International Conference on Machine Learning*, 2019. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:174800317) [org/CorpusID:174800317](https://api.semanticscholar.org/CorpusID:174800317).
- <span id="page-10-17"></span>Han, S., Pool, J., Tran, J., and Dally, W. J. Learning both weights and connections for efficient neural networks, 2015. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1506.02626) [1506.02626](https://arxiv.org/abs/1506.02626).
- <span id="page-10-13"></span>Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding, 2016. URL [https:](https://arxiv.org/abs/1510.00149) [//arxiv.org/abs/1510.00149](https://arxiv.org/abs/1510.00149).
- <span id="page-10-4"></span>He, S., Dong, D., Ding, L., and Li, A. Demystifying the compression of mixture-of-experts through a unified framework. *arXiv preprint arXiv:2406.02500*, 2024.
- <span id="page-10-18"></span>Hoang, D. N., Liu, S., Marculescu, R., and Wang, Z. REVIS-ITING PRUNING AT INITIALIZATION THROUGH THE LENS OF RAMANUJAN GRAPH. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=uVcDssQff_) [id=uVcDssQff\\_](https://openreview.net/forum?id=uVcDssQff_).
- <span id="page-10-9"></span>Hong, J., Duan, J., Zhang, C., Li, Z., Xie, C., Lieberman, K., Diffenderfer, J., Bartoldson, B., Jaiswal, A., Xu, K., et al. Decoding compressed trust: Scrutinizing the trustworthiness of efficient llms under compression. *arXiv preprint arXiv:2403.15447*, 2024.
- <span id="page-10-8"></span>Jaiswal, A., Gan, Z., Du, X., Zhang, B., Wang, Z., and Yang, Y. Compressing llms: The truth is rarely pure and never simple. *arXiv preprint arXiv:2310.01382*, 2023a.
- <span id="page-10-10"></span>Jaiswal, A., Liu, S., Chen, T., and Wang, Z. The emergence of essential sparsity in large pre-trained models: The weights that matter. *arXiv preprint arXiv:2306.03805*, 2023b.
- <span id="page-10-3"></span>Jaiswal, A., Yin, L., Zhang, Z., Liu, S., Zhao, J., Tian, Y., and Wang, Z. From galore to welore: How low-rank weights non-uniformly emerge from low-rank gradients. *arXiv preprint arXiv:2407.11239*, 2024.

- <span id="page-11-11"></span>Jaiswal, A. K., Ma, H., Chen, T., Ding, Y., and Wang, Z. Training your sparse neural network better with any mask. In *International Conference on Machine Learning*, pp. 9833–9844. PMLR, 2022.
- <span id="page-11-1"></span>Jaiswal, A. K., Liu, S., Chen, T., Ding, Y., and Wang, Z. Instant soup: Cheap pruning ensembles in a single pass can draw lottery tickets from large models. In *International Conference on Machine Learning*, pp. 14691– 14701. PMLR, 2023c.
- <span id="page-11-8"></span>Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-11-10"></span>Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020. URL [https://arxiv.org/abs/2001.](https://arxiv.org/abs/2001.08361) [08361](https://arxiv.org/abs/2001.08361).
- <span id="page-11-5"></span>Kim, J., Lee, J. H., Kim, S., Park, J., Yoo, K. M., Kwon, S. J., and Lee, D. Memory-efficient finetuning of compressed large language models via sub-4-bit integer quantization. *ArXiv*, abs/2305.14152, 2023. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:258841104) [org/CorpusID:258841104](https://api.semanticscholar.org/CorpusID:258841104).
- <span id="page-11-19"></span>Kim, Y. J., Awan, A. A., Muzio, A., Salinas, A. F. C., Lu, L., Hendy, A., Rajbhandari, S., He, Y., and Awadalla, H. H. Scalable and efficient moe training for multitask multilingual models, 2021. URL [https://arxiv.](https://arxiv.org/abs/2109.10465) [org/abs/2109.10465](https://arxiv.org/abs/2109.10465).
- <span id="page-11-20"></span>Koishekenov, Y., Berard, A., and Nikoulina, V. Memoryefficient nllb-200: Language-specific expert pruning of a massively multilingual machine translation model, 2023. URL <https://arxiv.org/abs/2212.09811>.
- <span id="page-11-12"></span>LeCun, Y., Denker, J., and Solla, S. Optimal brain damage. In Touretzky, D. (ed.), *Advances in Neural Information Processing Systems*, volume 2. Morgan-Kaufmann, 1989. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) [cc/paper\\_files/paper/1989/file/](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf) [6c9882bbac1c7093bd25041881277658-Paper](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf).
- <span id="page-11-2"></span>Lee, N., Ajanthan, T., and Torr, P. Snip: Single-shot network pruning based on connection sensitivity. In *International Conference on Learning Representations*, 2019. URL [https://openreview.net/forum?](https://openreview.net/forum?id=B1VZqjAcYX) [id=B1VZqjAcYX](https://openreview.net/forum?id=B1VZqjAcYX).
- <span id="page-11-21"></span>Li, P., Jin, X., Cheng, Y., and Chen, T. Examining posttraining quantization for mixture-of-experts: A benchmark, 2024a. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2406.08155) [2406.08155](https://arxiv.org/abs/2406.08155).

- <span id="page-11-14"></span>Li, P., Zhang, Z., Yadav, P., Sung, Y.-L., Cheng, Y., Bansal, M., and Chen, T. Merge, then compress: Demystify efficient smoe with hints from its routing policy, 2024b. URL <https://arxiv.org/abs/2310.01334>.
- <span id="page-11-13"></span>Li, Y., Du, X., Jaiswal, A., Lei, T., Zhao, T., Wang, C., and Wang, J. Idea prune: An integrated enlarge-andprune pipeline in generative language model pretraining. 2025. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:276903261) [org/CorpusID:276903261](https://api.semanticscholar.org/CorpusID:276903261).
- <span id="page-11-6"></span>Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S. Awq: Activation-aware weight quantization for llm compression and acceleration. *ArXiv*, abs/2306.00978, 2023. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:258999941) [org/CorpusID:258999941](https://api.semanticscholar.org/CorpusID:258999941).
- <span id="page-11-9"></span>Lin, S., Lyu, P., Liu, D., Tang, T., Liang, X., Song, A., and Chang, X. Mlp can be a good transformer learner. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 19489–19498, 2024.
- <span id="page-11-3"></span>Liu, S., Chen, T., Zhang, Z., Chen, X., Huang, T., Jaiswal, A., and Wang, Z. Sparsity may cry: Let us fail (current) sparse neural networks together! *arXiv preprint arXiv:2303.02141*, 2023a.
- <span id="page-11-15"></span>Liu, Z., Li, J., Shen, Z., Huang, G., Yan, S., and Zhang, C. Learning efficient convolutional networks through network slimming, 2017. URL [https://arxiv.org/](https://arxiv.org/abs/1708.06519) [abs/1708.06519](https://arxiv.org/abs/1708.06519).
- <span id="page-11-4"></span>Liu, Z., Oguz, B., Zhao, C., Chang, E., Stock, P., Mehdad, Y., Shi, Y., Krishnamoorthi, R., and Chandra, V. Llm-qat: Data-free quantization aware training for large language models. *arXiv preprint arXiv:2305.17888*, 2023b.
- <span id="page-11-7"></span>Lu, X., Liu, Q., Xu, Y., Zhou, A., Huang, S., Zhang, B., Yan, J., and Li, H. Not all experts are equal: Efficient expert pruning and skipping for mixture-of-experts large language models. *arXiv preprint arXiv:2402.14800*, 2024.
- <span id="page-11-18"></span>Mishra, A., Latorre, J. A., Pool, J., Stosic, D., Stosic, D., Venkatesh, G., Yu, C., and Micikevicius, P. Accelerating sparse deep neural networks, 2021. URL [https://](https://arxiv.org/abs/2104.08378) [arxiv.org/abs/2104.08378](https://arxiv.org/abs/2104.08378).
- <span id="page-11-0"></span>Mittal, S., Bengio, Y., and Lajoie, G. Is a modular architecture enough? *Advances in Neural Information Processing Systems*, 35:28747–28760, 2022.
- <span id="page-11-17"></span>Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz, J. Pruning convolutional neural networks for resource efficient inference, 2017. URL [https://arxiv.org/](https://arxiv.org/abs/1611.06440) [abs/1611.06440](https://arxiv.org/abs/1611.06440).
- <span id="page-11-16"></span>Molchanov, P., Mallya, A., Tyree, S., Frosio, I., and Kautz, J. Importance estimation for neural network pruning, 2019. URL <https://arxiv.org/abs/1906.10771>.

- <span id="page-12-2"></span>Muzio, A., Sun, A., and He, C. Seer-moe: Sparse expert efficiency through regularization for mixture-of-experts. *arXiv preprint arXiv:2404.05089*, 2024.
- <span id="page-12-8"></span>Nvidia. Nvidia a100 tensor core gpu architecture. *https://www.nvidia.com/content/dam/enzz/Solutions/Data-Center/nvidia-ampere-architecturewhitepaper.pdf*, 2020.
- <span id="page-12-16"></span>Paul, M., Chen, F., Larsen, B. W., Frankle, J., Ganguli, S., and Dziugaite, G. K. Unmasking the lottery ticket hypothesis: What's encoded in a winning ticket's mask?, 2022. URL <https://arxiv.org/abs/2210.03044>.
- <span id="page-12-14"></span>Rajbhandari, S., Li, C., Yao, Z., Zhang, M., Aminabadi, R. Y., Awan, A. A., Rasley, J., and He, Y. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale, 2022. URL [https://](https://arxiv.org/abs/2201.05596) [arxiv.org/abs/2201.05596](https://arxiv.org/abs/2201.05596).
- <span id="page-12-5"></span>Sanyal, A., Torr, P. H., and Dokania, P. K. Stable rank normalization for improved generalization in neural networks and gans. In *International Conference on Learning Representations*, 2020. URL [https://openreview.](https://openreview.net/forum?id=H1enKkrFDB) [net/forum?id=H1enKkrFDB](https://openreview.net/forum?id=H1enKkrFDB).
- <span id="page-12-17"></span>Sarkar, S., Lausen, L., Cevher, V., Zha, S., Brox, T., and Karypis, G. Revisiting smoe language models by evaluating inefficiencies with task specific expert pruning, 2024. URL <https://arxiv.org/abs/2409.01483>.
- <span id="page-12-0"></span>Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017a.
- <span id="page-12-10"></span>Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017b. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1701.06538) [1701.06538](https://arxiv.org/abs/1701.06538).
- <span id="page-12-15"></span>Shen, M., Yin, H., Molchanov, P., Mao, L., Liu, J., and Alvarez, J. M. Structural pruning via latency-saliency knapsack, 2022. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2210.06659) [2210.06659](https://arxiv.org/abs/2210.06659).
- <span id="page-12-7"></span>Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. *arXiv preprint arXiv:2306.11695*, 2023.
- <span id="page-12-12"></span>Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models, 2024. URL <https://arxiv.org/abs/2211.10438>.
- <span id="page-12-6"></span>Yin, L., Liu, S., Jaiswal, A., Kundu, S., and Wang, Z. Junk dna hypothesis: A task-centric angle of llm

- pre-trained weights through sparsity. *arXiv preprint arXiv:2310.02277*, 2023a.
- <span id="page-12-1"></span>Yin, L., Wu, Y., Zhang, Z., Hsieh, C.-Y., Wang, Y., Jia, Y., Pechenizkiy, M., Liang, Y., Wang, Z., and Liu, S. Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity. *arXiv preprint arXiv:2310.05175*, 2023b.
- <span id="page-12-3"></span>Zhang, Z., Jaiswal, A., Yin, L., Liu, S., Zhao, J., Tian, Y., and Wang, Z. Q-galore: Quantized galore with int4 projection and layer-adaptive low-rank gradients. *arXiv preprint arXiv:2407.08296*, 2024.
- <span id="page-12-13"></span>Zhangheng, L., Liu, S., Chen, T., Jaiswal, A. K., Zhang, Z., Wang, D., Krishnamoorthi, R., Chang, S., and Wang, Z. Sparse cocktail: Every sparse pattern every sparse ratio all at once. In *Forty-first International Conference on Machine Learning*.
- <span id="page-12-4"></span>Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., and Tian, Y. Galore: Memory-efficient llm training by gradient low-rank projection. *arXiv preprint arXiv:2403.03507*, 2024.
- <span id="page-12-9"></span>Zhou, A., Ma, Y., Zhu, J., Liu, J., Zhang, Z., Yuan, K., Sun, W., and Li, H. Learning n: m fine-grained structured sparse neural networks from scratch. *arXiv preprint arXiv:2102.04010*, 2021.
- <span id="page-12-11"></span>Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., and Fedus, W. St-moe: Designing stable and transferable sparse expert models. *arXiv preprint arXiv:2202.08906*, 2022.

### A. Related Work

SMoE and Its Superiority. It is widely acknowledged that scaling model size benefits performance by enhancing learning capacity and generalization ability [\(Brown](#page-9-2) [et al.,](#page-9-2) [2020;](#page-9-2) [Kaplan et al.,](#page-11-10) [2020\)](#page-11-10). To achieve more efficient model scaling, Sparsely activated Mixture-of-Experts (SMoE) [\(Shazeer et al.,](#page-12-10) [2017b;](#page-12-10) [Zoph et al.,](#page-12-11) [2022;](#page-12-11) [Du et al.,](#page-10-12) [2022\)](#page-10-12) has emerged as a widely adopted approach, enabling the training of larger models with negligible additional computational overhead [\(Jiang et al.,](#page-11-8) [2024;](#page-11-8) [Dai et al.,](#page-9-3) [2024;](#page-9-3) [DeepSeek-AI et al.,](#page-9-4) [2024\)](#page-9-4). Given the predominance of Transformer architectures in NLP, numerous research efforts have focused on incorporating MoE layers within the feed-forward neural networks of these models. In pursuit of enhanced SMoE models, various iterations of the standard MoE architecture have been proposed. For example, DeepSeek-MoE [\(Dai et al.,](#page-9-3) [2024;](#page-9-3) [DeepSeek-AI et al.,](#page-9-4) [2024\)](#page-9-4) utilizes a large number of finely segmented experts, designating a subset as shared experts to capture common knowledge. More recently, Mixtral [\(Jiang et al.,](#page-11-8) [2024\)](#page-11-8) has demonstrated that SMoE can achieve performance comparable to full-parameter LLMs while utilizing significantly fewer active parameters.

Compression for LLMs and SMoEs. LLMs have demonstrated remarkable success. However, their substantial memory and computational requirements pose deployment challenges. Numerous model compression techniques have been proposed to address this issue. Algorithmically, these methods can be classified into three main categories: 1 Quantization, which converts float32 weights or activations to lower-bit representations[\(Lin et al.,](#page-11-6) [2023;](#page-11-6) [Frantar et al.,](#page-10-2) [2022;](#page-10-2) [Jaiswal et al.,](#page-11-11) [2022;](#page-11-11) [Xiao et al.,](#page-12-12) [2024\)](#page-12-12); 2 Pruning, which eliminates less critical components, such as weights, neurons, or layers [\(LeCun et al.,](#page-11-12) [1989;](#page-11-12) [Li et al.,](#page-11-13) [2025;](#page-11-13) [Han](#page-10-13) [et al.,](#page-10-13) [2016;](#page-10-13) [Zhangheng et al.;](#page-12-13) [Sun et al.,](#page-12-7) [2023\)](#page-12-7); 3 Knowledge distillation, which transfers knowledge from a larger model to a smaller one [\(Gou et al.,](#page-10-14) [2021;](#page-10-14) [Li et al.,](#page-11-14) [2024b;](#page-11-14) [Rajbhandari et al.,](#page-12-14) [2022\)](#page-12-14). In this study, we concentrate on model pruning for compression, which is generally divided into *structured* and *unstructured* approaches. Structured pruning methods [\(Liu et al.,](#page-11-15) [2017;](#page-11-15) [Molchanov et al.,](#page-11-16) [2019;](#page-11-16) [Shen et al.,](#page-12-15) [2022;](#page-12-15) [Fang et al.,](#page-10-15) [2023\)](#page-10-15) eliminate entire structured components of a network, facilitating straightforward GPU acceleration. Existing techniques primarily rely on weight or activation statistics of neurons [\(Dubey et al.,](#page-10-16) [2018;](#page-10-16) [Bandari et al.,](#page-9-5) [2024;](#page-9-5) [Molchanov et al.,](#page-11-17) [2017\)](#page-11-17). Unstructured methods [\(Han et al.,](#page-10-17) [2015;](#page-10-17) [Paul et al.,](#page-12-16) [2022;](#page-12-16) [Hoang et al.,](#page-10-18) [2023\)](#page-10-18) operate at the individual weight level, preserving performance at higher sparsity levels but typically requiring additional effort to enable GPU speedups [\(Mishra et al.,](#page-11-18) [2021\)](#page-11-18).

SMoE architectures enable the scaling of LLMs but neces-

sitate substantial memory to host experts while exhibiting expert redundancy. To address these challenges, numerous studies have also focused on developing SMoE-modelspecific compression techniques. Initial approaches [\(Chen](#page-9-6) [et al.,](#page-9-6) [2022;](#page-9-6) [Kim et al.,](#page-11-19) [2021;](#page-11-19) [Koishekenov et al.,](#page-11-20) [2023;](#page-11-20) [Sarkar et al.,](#page-12-17) [2024\)](#page-12-17) propose expert pruning based on utilization metrics; however, these methods often resulted in diminished performance. Subsequent research [\(Rajbhan](#page-12-14)[dari et al.,](#page-12-14) [2022;](#page-12-14) [Fedus et al.,](#page-10-6) [2022;](#page-10-6) [Artetxe et al.,](#page-9-7) [2022\)](#page-9-7) explores the creation of smaller models, either dense or SMoE-based, with reduced layer counts through knowledge distillation (KD). While effective, this approach demands significant computational resources and fails to address the inherent redundancy among experts. More recently, MC-SMoE [\(Li et al.,](#page-11-14) [2024b\)](#page-11-14) dynamically merges experts during inference time, though it is limited to specific tasks. Besides pruning-based methods, there are also a few works that specifically study quantization in SMoE models [\(Li et al.,](#page-11-21) [2024a\)](#page-11-21).

### <span id="page-13-0"></span>B. Training Duration and MoE Lottery Networks

MoE lottery subnetworks rely on *estimate-prune-finetune* procedure to mitigate the abrupt impact of expert dropping of the resultant subnetwork. More specifically, finetuning routine using pre-training objectives helps in balancing expert load distribution and performance improvement. One natural question that arises is: *Given the enormous computational cost of finetuning SMoEs, how much finetuning will be sufficient to achieve a reasonable performance gain facilitated by it?*

| Training Tokens →     | 0.25M | 0.51M | 1.13M | 2.27M |
|-----------------------|-------|-------|-------|-------|
| Mixtral 8×7B          | 13.55 | 13.51 | 13.05 | 13.01 |
| Mixtral 8×7B Instruct | 14.82 | 14.19 | 14.02 | 14.08 |

<span id="page-13-1"></span>Table 5. Performance comparison (perplexity) wrt. total training tokens used in task-agnostic finetuning of Mistral checkpoints with 75% expert dropping.

Table [5](#page-13-1) presents the performance (perplexity) of Mixtral-7×8B Base and Instruct model checkpoints when 6 out of 8 experts are dropped from every layer using the Minimum Expert Activation Norm (Min-EAN) criterion. Each column in Table [5](#page-13-1) indicates the total number of training tokens used during the finetuning subroutine of the MoE Lottery Subnetwork. It can be clearly observed that the benefits of task-agnostic finetuning saturates after a certain amount of training tokens. More specifically, we found that ∼1 million training tokens are sufficient to address the abrupt impact created by expert dropping and any additional finetuning brings marginal or no gain in performance.

### C. Additional Experimental Setup

| Hyperparameter             | CommonsenseQA | WinoGrande    | MMLU          | ARC-Easy     | BoolQ        |
|----------------------------|---------------|---------------|---------------|--------------|--------------|
| Train Samples (avg. words) | 9741(28.00)   | 63238 (39.96) | 1531 (84.97)  | 2247 (48.16) | 9427 (14.81) |
| Test Samples (avg. words)  | 1221(27.75)   | 1267(40.20)   | 14042 (84.28) | 2372 (48.42) | 3270 (14.70) |
| Batch Size                 | 8             | 8             | 4             | 8            | 8            |
| Max_length                 | 512           | 512           | 512           | 512          | 512          |
| Training Steps             | 2500          | 2500          | 1000          | 1500         | 2500         |
| Learning Rate              | 0.0001        | 0.0001        | 0.0001        | 0.0001       | 0.0001       |

<span id="page-14-0"></span>*Table 6.* Hyperparamters settings for zero-shot downstream fine-tuning of Mistral-8×7B models.

Our experiments are conducted on Mixtral MoE Base and Instruct downloaded from HuggingFace. For activation and gradient criterions, we propose to use a taskagnostic calibration C4 validation set of 256 samples with max\_seq\_len of 2048. As suggested in Table 5, the benefits of task-agnostic finetuning saturates with no significant benefits of prolonged finetuning, we propose a progressive scheduler for number of training tokens required for k rounds of MoE lottery pruning to miminize compute requirements. More specifically, we double the amount of tokens every round starting from 0.2M tokens for first round. We used adamw with a cosine learning scheduler with maximum learning rate of 1e - 6. With the availability of 8×A100, we use a batch size of 8 and every round we reset the optimizer. Additional details for our downstream finetuning tasks are provided in Table 6 and we followed the exactly same settings for all compression level.

## D. Performance comparison with SoTA MoE Expert Pruning Methods

| Method               | Total Expert Sparsity(\u00e7) | Accuracy Drop from Dense $(\downarrow)$ | Memory $Usage(\downarrow)$ | Speedup(↑) |
|----------------------|-------------------------------|-----------------------------------------|----------------------------|------------|
| Dense                | 0                             | 0                                       | ×1                         | ×1         |
| Random               | 50%                           | 20.46                                   | ×0.55                      | ×1.27      |
| (Lu et al., 2024)    | 50%                           | 14.38                                   | ×0.55                      | ×1.27      |
| (Muzio et al., 2024) | 50%                           | 13.78                                   | ×0.55                      | ×1.27      |
| Ours                 | 50%                           | 13.05                                   | ×0.55                      | ×1.27      |

Table 7. Comparison with baseline approaches. MC-Suite Criterion (Min-EAN) achieves the minimal accuracy drop from the dense baseline at all expert sparsity levels. For (Muzio et al., 2024) we use the numbers reported in the paper due to unavailability of code to reproduce.

### E. MoE Experts and MC-Suite Criterions

![](_page_14_Figure_9.jpeg)

Figure 8. Experts Vocabulary Coverage Criterion (EVC): Illustration of experts vocabulary coverage corresponding to different MoE layers from Mixtral-8×7B Base model. Experts with minimum vocabulary coverage are better candidates for dropping.

![](_page_14_Figure_11.jpeg)

Figure 9. Experts-Usage Frequency (EUF): Expert usage frequency indicate how frequently an expert e is activated and above heatmap indicate experts from different MoE layers from Mixtral-8×7B Base model. Interestingly, it can be observed that there multiple experts with significantly low expert usage making them good candidate for expert dropping.

![](_page_15_Figure_1.jpeg)

Figure 10. Experts-Expert Collaboration (ECC): Snapshot of Expert-Expert Collaboration estimated using C4 dataset for Mixtral-8×7B Base model. Least dominant expert are identified as expert which have highest collaboration with rest of other experts within corresponding layer.

![](_page_15_Figure_3.jpeg)

Figure 11. Expert Input Token Similarity (ETS): Snapshot of Expert-Expert Input token similarity estimated using C4 dataset for Mixtral-8×7B Base model. Higher level of input token similarity indicate existence of redundancy and can be used as a signal to identify least dominant expert.

![](_page_16_Figure_1.jpeg)

Figure 12. Experts Activation Similarity (EAS): Snapshot of Expert-Expert Activation similarity estimated using C4 dataset for Mixtral-8×7B Base model. Least dominant expert are identified as expert which have highest similarity with rest of other experts within corresponding layer.

![](_page_16_Figure_3.jpeg)

Figure 13. Experts Activation Entropy (EAE): Heatmap corresponding to activation entropy estimated for different experts using C4 dataset for Mixtral-8×7B Base model. Interesting, we find that activation entropy gradually increases as we move from intial layers to terminal MoE layers. Experts with minimal activation entropy within a MoE layer are better candidates for dropping. Note that even in some initial layers, it can be observed that some experts carry notable entropy and dropping them lead to significant performance degradation.

![](_page_16_Figure_5.jpeg)

Figure 14. Experts Gradient Entropy (EGE): Illustration of the gradient entropy estimated using C4 dataset for Mixtral-8×7B Base model. We found a strong positive co-relation between the experts with high activation entropy and gradient entropy. Similar to activation entropy, we found two experts in Layer 1 and 2 of the checkpoint having significantly high gradient rntropy and dropping them lead to abrupt performance drop.

![](_page_16_Figure_7.jpeg)

Figure 15. Experts Weight Similarity (EWS): Heatmap illustrating the weigh similarity acorss 8 experts corresponding to 32 MoE layers of Mixtral-8×7B Base model. Expert with highest weight similarity across remaining 7 experts becomes the better candidate for expert dropping.