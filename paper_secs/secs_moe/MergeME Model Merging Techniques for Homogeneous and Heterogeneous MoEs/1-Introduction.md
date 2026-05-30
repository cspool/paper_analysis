# 1 Introduction

Large language models (LLMs) pretrained on a wide-variety of corpora have achieved notable success in multiple tasks [\(Touvron et al.,](#page-10-0) [2023;](#page-10-0) [Ope](#page-10-1)[nAI,](#page-10-1) [2023;](#page-10-1) [Brown et al.,](#page-9-0) [2020;](#page-9-0) [Liu et al.,](#page-10-2) [2024a\)](#page-10-2). With significant progress, there is increasing interest in how to continuously improve the performance of LLMs in new domains, including math [\(Yu et al.,](#page-11-0) [2023\)](#page-11-0), code [\(Roziere et al.,](#page-10-3) [2023\)](#page-10-3), Wikipedia knowledge [\(Shao et al.,](#page-10-4) [2024\)](#page-10-4), or legal domains [\(Cui et al.,](#page-9-1) [2023\)](#page-9-1). One straightforward approach is through continual pretraining (CPT) on domain-specific data, which, however, is challenging for multiple target domains, as it can

cause catastrophic forgetting on previously learned tasks [\(Luo et al.,](#page-10-5) [2023\)](#page-10-5).

An alternative approach is Mixture-of-Experts (MoE) merging, where dense experts are first CPTed in parallel for each domain and then merged into a unified MoE model, usually by keeping feedforward neural network (FFN) layers separate and averaging non-FFN layers [\(Sukhbaatar et al.,](#page-10-6) [2024;](#page-10-6) [Kang et al.,](#page-9-2) [2024\)](#page-9-2). Compared with dense models of similar size, the MoE model uses just a subset of parameters during inference by learning to route tokens to the top few experts, thus reducing inference costs. Unlike training an MoE model from scratch, MoE merging offers modularity, as individual experts are domain-specialized, and is substantially less expensive, as CPT-ing experts in parallel requires less compute than training the entire MoE on large datasets from the beginning [\(Sukhbaatar](#page-10-6) [et al.,](#page-10-6) [2024\)](#page-10-6).

In this paper, we investigate how to effectively merge different domain expert models into a unified MoE model. The current state-of-the-art (SoTA) MoE merging approach, such as Branch-Train-Mix (BTX) [\(Sukhbaatar et al.,](#page-10-6) [2024\)](#page-10-6) assumes experts are branched from the same ancestor model and merges experts by simply unweighted averaging the non-FFN layers. However, as experts diverge in the parameter space, for example by branching from different ancestors or by training on aggressively different data, unweighted averaging may not effectively handle parameter interference such as sign conflicts [\(Yu et al.,](#page-11-1) [2024;](#page-11-1) [Yadav et al.,](#page-11-2) [2024\)](#page-11-2). As a result, the merged MoE may underperform and will require a significant amount of additional fine-tuning to recover in performance, which is both expensive and could be impractical when the experts' training data is not publicly available. Furthermore, existing MoE merging methods cannot be directly used to merge heterogeneous experts with different architectures, which could be the case in practice, as increasingly more experts

are provided by separate teams, such as CodeLlama [\(Roziere et al.,](#page-10-3) [2023\)](#page-10-3) and Olmo [\(Groeneveld](#page-9-3) [et al.,](#page-9-3) [2024\)](#page-9-3). Therefore, it is still an open question how to effectively merge homogeneous and heterogeneous experts into an MoE combining the benefits of each.

To enable the use of diverse expert models, our work addresses the above limitations via new MoE merging methodologies for both homogeneous and heterogeneous experts. In summary, our work introduces three main contributions:

- We utilize advanced merging methods that address parameter interference, demonstrating their superiority over unweighted averaging in homogeneous expert merging, particularly in scenarios with limited resources for post-merging MoE fine-tuning.
- We propose a perplexity-based heuristic for routing token sequences to domain-specific experts in low-resource environments where MoE finetuning is not feasible.
- We develop a novel approach to merge experts with different architectures into a single MoE, which learns to route token sequences dynamically to the appropriate expert.

Through extensive experiments and ablation studies across benchmarks in mathematical reasoning, programming, and general knowledge, we show that our proposed methodologies outperform previous state-of-the-art methods and extend the practical applications of MoE merging.

### 2 Background and Related Work

### 2.1 Dense Model Merging

Dense merging methods combine multiple dense models into one to achieve diverse capabilities [\(Wortsman et al.,](#page-10-7) [2022;](#page-10-7) [Ilharco et al.,](#page-9-4) [2022;](#page-9-4) [God](#page-9-5)[dard et al.,](#page-9-5) [2024;](#page-9-5) [Jin et al.,](#page-9-6) [2022;](#page-9-6) [Matena and Raffel,](#page-10-8) [2022;](#page-10-8) [Roberts et al.,](#page-10-9) [2024\)](#page-10-9). Most approaches focus on merging homogeneous dense models into another dense model. For example, average merging [\(Wortsman et al.,](#page-10-7) [2022\)](#page-10-7) averages model parameters, while task vector merging [\(Ilharco et al.,](#page-9-4) [2022\)](#page-9-4) adds the unweighted sum of task vectors (the difference between base and expert parameters) back to the dense model with scaling. Other work determines task vector weights instead of using an unweighted sum [\(Jin et al.,](#page-9-6) [2022;](#page-9-6) [Matena](#page-10-8)

[and Raffel,](#page-10-8) [2022\)](#page-10-8). SoTA methods like Dare and Ties [\(Yadav et al.,](#page-11-2) [2024;](#page-11-2) [Yu et al.,](#page-11-1) [2024\)](#page-11-1) trim the task vector to resolve parameter interference: Dare trims the task vector randomly and rescales, while Ties sets vector parameters to zero by magnitude and adjusts signs to reduce conflicts.

In addition to homogeneous model merging, [Roberts et al.](#page-10-9) [\(2024\)](#page-10-9) propose merging heterogeneous models into a dense model using projectors, while [Wan et al.](#page-10-10) [\(2024\)](#page-10-10) apply knowledge distillation to fuse heterogeneous models. In this work, we introduce a more efficient method for merging experts with limited or no further fine-tuning and, unlike previous work focusing on dense models, we explore merging homogeneous and heterogeneous experts into an MoE model.

### 2.2 MoE Training and Merging

MoE architectures enable quicker inference with a certain parameter count by introducing Sparse MoE layers, where a router mechanism assigns tokens to the top-K expert FFNs (usually 1 or 2) in parallel [\(Fedus et al.,](#page-9-7) [2022;](#page-9-7) [Shazeer et al.,](#page-10-11) [2017;](#page-10-11) [Zhang et al.,](#page-11-3) [2022\)](#page-11-3). Most MoE training approaches, known as upcycling, train the entire model from scratch to handle multiple tasks [\(Komatsuzaki et al.,](#page-9-8) [2022;](#page-9-8) [Jiang et al.,](#page-9-9) [2024;](#page-9-9) [Dou et al.,](#page-9-10) [2024;](#page-9-10) [Dai](#page-9-11) [et al.,](#page-9-11) [2024\)](#page-9-11). These methods first initialize the MoE model from a pretrained base model and then train it on the entire dataset. However, due to the costly communication between GPUs, the upcycling method introduces significant computational overhead [\(Sukhbaatar et al.,](#page-10-6) [2024;](#page-10-6) [Li et al.,](#page-10-12) [2024b\)](#page-10-12). To address this, methods like Branch-Train-Merge (BTM) [\(Gururangan et al.,](#page-9-12) [2023;](#page-9-12) [Li et al.,](#page-10-13) [2022\)](#page-10-13) average model outputs from different experts, while Branch-Train-Mix (BTX) [\(Sukhbaatar et al.,](#page-10-6) [2024\)](#page-10-6) branches the base model, trains each on different domains, and merges them into a unified MoE. BTX is shown to be more effective than BTM as well as dense CPT and MoE upcycling baselines. Another recent approach, Self-MoE [\(Kang](#page-9-2) [et al.,](#page-9-2) [2024\)](#page-9-2), uses low-rank adaptation (LoRA) [\(Hu](#page-9-13) [et al.,](#page-9-13) [2021\)](#page-9-13) to fine-tune experts on generated synthetic data [\(Liu et al.,](#page-10-14) [2024b\)](#page-10-14) and combines trained adapters into an MoE. To our knowledge, we are the first to introduce a framework for merging heterogeneous models into an MoE.

<span id="page-2-3"></span>![](_page_2_Figure_0.jpeg)

Figure 1: Overview of the proposed MoE framework for homogeneous model merging. We replace averaging with Dare or Ties merging to reduce parameter interference. Additionally, we introduce novel routing heuristics to enhance performance without fine-tuning.

### 3 Methodology

We define our research problem as follows: Given l dense expert models with parameters  $[\theta_1,\theta_2,\ldots,\theta_l]$ , each pretrained on different domains, we aim to propose an efficient merging method to combine these dense models into an MoE with parameters  $\theta_m = \text{Merge}(\theta_1,\theta_2,\ldots,\theta_l)$ , optimizing performance across all domains.

We now present our approach for MoE merging with homogeneous and heterogeneous expert models. First, for MoE merging with homogeneous experts (Section 3.1), we propose replacing existing averaging with more advanced merging methods to deal with parameter interference, and introduce sequence-level routing heuristics to enhance MoE performance without post-merge fine-tuning. Second, we introduce a novel framework for MoE merging with heterogeneous experts (Section 3.2), which uses projectors to unify expert inputs and outputs, and a sequence-level router.

#### <span id="page-2-0"></span>3.1 Homogeneous Model Merging

First, we describe the basic merging setup (Section 3.1.1) and then summarize our extensions to resolve parameter interference (Section 3.1.2) and address the need for MoE fine-tuning (Section 3.1.3). The overall pipeline is visualized in Figure 1.

#### <span id="page-2-1"></span>3.1.1 Merging Setup

Our merging setup is similar to the BTX (Sukhbaatar et al., 2024), where it merges all non-FFN layers (embedding, attention, normalization, and head) of experts by unweighted

averaging and keeps the FFNs separate. As in standard MoE architectures, a router network, implemented as a Multilayer Perceptron (MLP), is inserted between the attention and FFN layers for token-level routing, selecting the top K (usually 1 or 2) experts for each layer among all l experts. The output of FFN layers  $\mathrm{FF}_{MoE}(v)$  of token embedding v is formulated as:

$$FF_{MoE}(v) = \sum_{i=1}^{K} SoftMax(top-K(\theta_r v))FF_i(v)$$

where  $\theta_r$  is the parameter of the router network and  $FF_i(v)$  is the output of each FFN experts for token v. After merging experts into a single MoE, BTX fine-tunes all parameters, including the router parameters on a mix of training data from all experts.

#### <span id="page-2-2"></span>3.1.2 Addressing Parameter Interference

The major pitfall of the unweighted merging is that there exists parameter interference, as explored in the previous work on dense model merging (Yu et al., 2024; Yadav et al., 2024). As suggested in Figure 2, when influential parameters (large magnitude parameters) in the task vector merge with redundant parameters (small magnitude parameters) or parameters with sign conflict, simple averaging will output a small magnitude parameter, which may reduce the effect of the original task vector.

<span id="page-2-4"></span>![](_page_2_Picture_14.jpeg)

Figure 2: Different types of parameter interference and merged outputs produced by simple averaging.

In contrast to BTX, we mitigate model interference by employing previous SoTA methods in this MoE setup, namely Dare and Ties. First, we calculate the task vector  $\tau_i = \theta_b - \theta_i$  with the base model parameter  $\theta_b$  and the parameter  $\theta_i$  for the model CPTed on domain i. For Ties merging, we first drop the bottom (100-p)% of the redundant parameters (smallest magnitude) by resetting them to 0. For each parameter, we determine the sign with the highest total magnitude in all task vectors and sum all task vectors together to  $\tau_m$  but only by keeping the parameter values whose signs are the

same as the determined sign. For Dare merging, we randomly drop the (100-p)% parameters. We rescale each task vector with  $\tau_i = \frac{\tau_i}{0.01p}$ . We sum all task vectors to  $\tau_m$ . Finally, we add the summed task vector back to the base model with the scaling term  $\lambda$  and obtain the merged layer parameters:  $\theta_m = \theta_b + \lambda \cdot \tau_m$ . We expect that the drop operation in both methods will address the parameter interference issue, as revealed in dense model merging, and produce a consistent performance boost (Yu et al., 2024; Yaday et al., 2024).

Similar to BTX, after combining each expert model into an MoE, we fine-tune all parameters in the MoE in the fine-tuning stage. By addressing parameter interference, our approach achieves performance improvements over BTX especially in earlier stages of fine-tuning. Next, we describe how to further reduce the fine-tuning needs.

#### <span id="page-3-1"></span>3.1.3 Reducing Fine-Tuning Needs

Fine-tuning MoEs is expensive due to the communication cost between GPUs (Sukhbaatar et al., 2024). Previous MoE merging methods require substantial fine-tuning of the MoE parameters to train the router network. In this section, we propose two techniques to reduce reliance on MoE fine-tuning, namely a perplexity-based routing and separating the attention layers.

The overall MoE pipeline after merging is illustrated in Figure 1, but we replace the router network with our routing heuristic to determine the expert selection. Additionally, we separate attention layers without merging them. For each input, the routing heuristic selects the appropriate experts and assigns their weights. The input is then processed by the chosen experts, and their outputs are combined using weights.

**Routing Heuristics** Our goal is to develop routing heuristics that replace the routing network without accessing the training data. We propose a sequence-level heuristics: perplexity (PPL) routing with only access to the inference sentence.

Our approach assesses the confidence of expert models by utilizing perplexity (PPL) to estimate their uncertainty. We then select the experts with the lowest PPL values, indicating higher confidence (Jelinek et al., 1977). Formally, with the inference input  $x_{inf}$  with t tokens and the expert parameter  $\theta_i$  for expert i, we compute the PPL value  $PPL(x_{inf}, \theta_i)$  as below:

$$PPL(x_{inf} \mid \theta_i) = \exp\left(-\frac{1}{t} \sum_{j=1}^{t} \log P(x_j \mid x_{< j}, \theta_i)\right)$$

where  $P(x_j \mid x_{< j}, \theta_i)$  is the probability assigned by model  $\theta_i$  on j-th token, given previous tokens.

Since a higher PPL indicates greater uncertainty, we use the reciprocal of PPL values to represent the model's confidence. With the top-K routing, the selected experts and their weights  $\alpha$  can be computed as follows:

$$\alpha = \text{SoftMax}(\text{top-K}(\frac{1}{\text{PPL}(x_{inf}|\theta_1)}, \dots, \frac{1}{\text{PPL}(x_{inf}|\theta_l)}))$$

Additionally, we also propose another routing heuristic based on the task vector and we present the details of this heuristic in Appendix C. With the routing heuristics and the corresponding computed weights from the heuristic, we will present the detailed merging process to form the MoE without further fine-tuning.

Separating attention layers We hypothesize that by merging attention layers, BTX creates inconsistency between the attention and FFN outputs. Specifically, the merged attention layers are influenced by all l task vectors from the dense experts, while the top-k routing method limits the FFN output to only k task vectors, leading to mismatched outputs. To address this, we consider keeping experts' attention layers as separate, similar to FFN. This ensures that both the attention and FFN layers come from the same expert, eliminating discrepancies from inconsistent task vector counts.

#### <span id="page-3-0"></span>3.2 Heterogeneous Model Merging

This section describes how to merge models with different architectures into a unified MoE. Previous MoE merging techniques cannot be directly used in this setting, as it is not possible to merge non-FFN networks layer by layer when experts have different numbers of layers or different layer shapes. To resolve this challenge, we propose a new merging method, which introduces projector layers and sequence-level routing as shown in Figure 3.

First, we denote the hidden dimension of all l experts as  $d_1, d_2 \ldots, d_l$ , and the maximum dimension among them is  $d_m$ . Suppose that we have a vocabulary  $\mathcal V$  and an input sentence with tokens  $[v_1, v_2 \ldots, v_t]$ . For the shared embedding layer  $\mathcal M_e$ , it maps the token  $v_i$  in the sentence to embedding  $e_i \in \mathbb R^{d_m}$  and the shared head layer is the network  $\mathcal M_h: \mathbb R^{d_m} \to \mathbb R^{|\mathcal V|}$ , which maps the weighted sum of projectors back to the probability distribution of tokens in the vocabulary. The

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 3: Overview of the proposed MoE framework for heterogeneous experts. Each color represents one heterogeneous expert.  $n_1, \dots, n_4$  refers to the number of layers in each expert.

embedding and head layer parameters are initialized from an averaging of the embedding and head layers of each expert. For experts with a hidden dimension less than  $d_m$ , we add padding zeros for their embedding and head layers before averaging.

Since we do not merge attention layers due to heterogeneous experts, all tokens must be routed to the same expert. Otherwise, the attention layers cannot perform self-attention, as they require access to every token. Hence, we average the token embeddings and use the router to perform the sequence-level routing. Formally, for top-K routing with router parameters  $\theta_r$ , the router computes the model weights as follows:

$$\alpha = \text{SoftMax}(\text{top-K}(\theta_r \text{avg}(e_1, e_2, \dots, e_t)))$$

For projectors: Proj-in and Proj-out, for each expert, randomly initialized MLP layers, they project the embedding outputs to the dimension of each expert, and project the expert output back to the maximum dimension. For *i*-th expert, we define:

Proj-in layer : 
$$\mathbb{R}^{d_m} \to \mathbb{R}^{d_i}$$
, Proj-out layer :  $\mathbb{R}^{d_i} \to \mathbb{R}^{d_m}$ 

After using the selected K experts to process the input sequences and translating their outputs to the representation  $r_i$  via the Proj-out layer (with dimension  $d_m$ ), we combine the representations using the router's weights:  $\sum_{i=1}^{K} \alpha_i r_i$ . The combined representation is then fed into the head layer to obtain the token probabilities.

After merging the heterogeneous experts into the MoE model, we choose an arbitrary tokenizer from

one expert, following previous work (Roberts et al., 2024) and fine-tune all parameters.

### 4 Experiments Setup and Model Analysis

Through our extensive empirical analysis, we aim to evaluate our frameworks in the settting of homogeneous experts and heterogeneous experts.

#### 4.1 Evaluation Dataset

We evaluate our proposed methodology on 6 datasets from three domains, as in the previous work (Sukhbaatar et al., 2024). For math reasoning, we choose GSM8K (8-shot) and MATH (4-shot) (Cobbe et al., 2021; Hendrycks et al., 2021). For code generation, we choose MBPP (0-shot) and HumanEval (0-shot) (Chen et al., 2021; Austin et al., 2021). For world knowledge, we choose Natural Questions (NQ, 5-shot) and TriviaQA (5-shot) (Kwiatkowski et al., 2019; Joshi et al., 2017).

#### <span id="page-4-1"></span>4.2 Model Configuration

This section describes the base model and experts discussed in our experiments:

- Base Model (Base-1B): This is our base model with 1B parameters and Llama-like architecture. We pretrain Base-1B from scratch with 250 billion (250B) tokens from the following datasets from the RedPajama dataset (Together Computer, 2023): Arxiv, CommonCrawl, C4, Stack-Exchange data and the first half of the WikiPedia data in the RedPajama dataset.
- Math Expert: We CPT the Base model on the OpenWebMath data for 100B tokens (Paster et al., 2023).
- **Code Expert**: We use the GitHub data in RedPajama to CPT the Base model for 100B tokens.
- **Knowledge Expert**: We CPT the Base-1B model on the second half of the Wikipedia data in the RedPajama dataset for 100B tokens.
- Math TinyLlama and Math Olmo: We CPT the TinyLlama-1.1B model (Zhang et al., 2024) and Olmo-1B model (Groeneveld et al., 2024) on the same data mixture of the Math Expert.
- Mixture of Experts (MoE): For homogeneous model merging, we combine three experts (Math Expert, Code Expert, Knowledge Expert) and one base model (Base-1B) into an MoE. For heterogeneous merging, we combine Code Expert,

Knowledge Expert, Base-1B, and either Math TinyLlama or Math Olmo. MoE fine-tuning is performed on all data sources from the base and expert models, using an additional 40B tokens. Detailed sampling ratios for pretraining and fine-tuning are provided in Appendix B.

We present the details of model architecture for each expert in Appendix A.

#### <span id="page-5-1"></span>4.3 Baseline Methods

To demonstrate the effectiveness of our methodology, we compare the performance of the merged 4-expert MoE models with several other baselines.

- Base & Experts: The dense base and expert models in Section 4.2.
- **BTX** (Sukhbaatar et al., 2024): The MoE model derived from the BTX pipeline with average merging and post-merge fine-tuning.
- **Random Routing**: The average merged MoE with randomly initialized router.
- **Router Fine-tuning**: The MoE model derived from the BTX pipeline but only fine-tune the parameters in the router network.
- 3-expert MoE: To demonstrate the functionality of Math Olmo or TinyLlama in heterogeneous expert merging, we prepare 3-expert MoE models (Base, Knowledge Expert, Code Expert), finetuned either on the full data source (including math) or only on code- and knowledge-related data. We merge these models using the BTX method, naming them 3-expert MoE (same data) and 3-expert MoE (w/o math).
- Dare Dense (Yu et al., 2024), Ties Dense (Yadav et al., 2024): Advanced dense model merging method. We apply Dare or Ties to merge four LMs to one dense model.

The details of the model configuration of the baseline methods are included in Appendix A.

#### 4.4 Similarity of Model Parameters

Before presenting the performance of our proposed methodology, we first analyze the similarities in model parameters across different experts to demonstrate the necessity for alternatives to average merging. Previous work assumes that parameters in attention layers are less domain-specialized,

leading to the use of simple averaging when combining non-FFN layers (Sukhbaatar et al., 2024). Our analysis aims to verify whether this assumption holds true for experts trained on different domains.

To quantify the degree of domain specialization in the model layers, we first extract the task vectors for each layer from our Math and Code Expert models. We then concatenate the task vectors from the attention layers and FFNs into two long vectors. Next, we calculate the cosine similarity between the two concatenated task vectors. The cosine similarity for the task vectors of the FFNs and self-attention layers is visualized separately in Figure 4.

<span id="page-5-0"></span>![](_page_5_Figure_15.jpeg)

Figure 4: Similarity of task vector for attention and FFNs layers for Math and Code Expert experts. We average the similarity of attentions or FFNs in one decoder layers as the overall similarity for each layer.

We observe that the task vectors from both layers exhibit low similarity, suggesting that the assumption of similar attention layers does not consistently hold and parameter interference may occur. This analysis demonstrates the need for more advanced merging methods, rather than averaging, for homogeneous model merging.

#### 5 Results

### 5.1 Homogeneous Model Merging

#### <span id="page-5-2"></span>5.1.1 Averaging vs. Dare / Ties

Replacing simple averaging with Dare or Ties merging obtains better performance. In this section, we demonstrate the superiority of our proposed Ties and Dare merging MoE over the BTX merging method. We present the performance of MoE models with **Dare merging** or **Ties merging** on non-FFN layers and other baselines in Table 1.

The details of training cost for each method are presented in Table 6 in Appendix.

<span id="page-6-0"></span>

| Method             | MBPP             | HumanEval | MATH | GSM8K | NQ   | TriviaQA | Avg.  |
|--------------------|------------------|-----------|------|-------|------|----------|-------|
| Dense Model        |                  |           |      |       |      |          |       |
| Base-1B            | 4.60             | 3.04      | 2.42 | 1.44  | 6.61 | 26.72    | 7.47  |
| Code Expert        | 10.2             | 8.53      | 2.42 | 2.57  | 3.11 | 16.70    | 7.26  |
| Math Expert        | 9.80             | 6.71      | 7.81 | 6.36  | 5.48 | 19.86    | 9.34  |
| Knowledge Expert   | 3.60             | 4.26      | 2.62 | 2.04  | 5.65 | 28.71    | 7.81  |
| MoE Merging        |                  |           |      |       |      |          |       |
| Random Routing     | 4.00             | 6.10      | 2.78 | 2.05  | 4.86 | 21.75    | 6.92  |
| Router Fine-tuning | 3.60             | 6.71      | 2.42 | 2.96  | 5.82 | 25.98    | 7.92  |
| BTX merging        | 12.40            | 11.58     | 6.74 | 7.73  | 6.78 | 25.10    | 11.72 |
| Ties merging       | 14.20            | 11.98     | 6.74 | 7.81  | 6.72 | 27.66    | 12.52 |
| Dare merging       | 14.20            | 10.98     | 6.82 | 7.96  | 6.50 | 30.68    | 12.86 |
|                    | MoE from Scratch |           |      |       |      |          |       |
| MoE Upcycling      | 18.40            | 12.20     | 7.80 | 12.21 | 8.37 | 37.33    | 16.05 |

Table 1: **Performance of proposed Dare and Ties merged MoE and other baselines across six datasets.** The best performance of Dense and MoE model is marked in bold. Results of Dare and Ties merged MoE outperform the BTX MoE and other baseline methods.

From Table 1, we see that individual experts generally achieve the best performance in their respective domains, as expected. However, CPTed Expert models experience catastrophic forgetting. For instance, both Code and Math Expert perform worse than Base-1B on the TriviaQA and NQ datasets.

The results in Table 1 show that using Ties or Dare merging significantly improves MoE performance over the BTX pipeline across almost all datasets, with a relative improvement of 6.94% and 9.72% in average performance. This suggests that advanced merging methods reduce weight interference and enhance performance.

As a reference, we include the results of MoE sparse upcycling (Komatsuzaki et al., 2022) in the last row of Table 1. This approach initializes the MoE model by creating four identical copies of the FFN layers from the base model and then CPT on the same 340B tokens used in our pipeline. However, we do not directly compare our results with the upcycling method, as it involves pretraining the entire MoE on all data, incurring significantly higher costs. We also visualize the average performance for each merging method with different finetuning token numbers in Figure 10 in Appendix D. In Figure 10, we observe that the Dare and Ties merging MoE models consistently outperform the BTX merging MoE throughout fine-tuning, especially in the earlier stages of fine-tuning.

MoE with Dare or Ties merging routes more tokens to domain experts. To further explore the effectiveness of Dare and Ties merging MoE, we evaluate MoEs on multiple benchmarks and calculate the routing probability averaged from each layer and token. We visualize the routing probabil-

<span id="page-6-1"></span>![](_page_6_Figure_7.jpeg)

![](_page_6_Figure_8.jpeg)

Figure 5: Routing probability of experts on GSM8K and MATH for different merging methods.

ity of each method of two math datasets (MATH and GSM8K) in Figure 5 and for other datasets, we put the results in Figure 7 in Appendix D.

Compared to MoEs with BTX merging, where the base model accepts the most routing decisions, the Dare and Ties merging method routes tokens to domain experts more frequently, as suggested in Figure 5. For example, for the GSM8K dataset, the routing probability for math expert increases from 0.28 to 0.35 or 0.46 when replacing simple averaging with the Ties or Dare merging. This finding suggests that the more effective MoE with the more advanced merging method should be attributed to more optimized routing decisions.

#### **5.1.2** Merging without Fine-tuning

In this part, we will evaluate our proposed routing heuristics in Section 3.1.3 for MoE without fine-tuning. Before we evaluate the overall performance of each benchmark, we will first examine the routing decision with our proposed heuristics. We present the routing probability for PPL routing heuristics for each dataset in Table 2.

<span id="page-6-2"></span>

| Benchmark | Base | Code | Math | Knowledge |
|-----------|------|------|------|-----------|
| GSM8K     | 23%  | 2%   | 43%  | 32%       |
| MATH      | 22%  | 2%   | 49%  | 27%       |
| MBPP      | 19%  | 22%  | 44%  | 15%       |
| HumanEval | 5%   | 43%  | 45%  | 7%        |
| NQ        | 43%  | 4%   | 10%  | 43%       |
| TriviaQA  | 50%  | 0%   | 0%   | 50%       |

Table 2: Routing probability of PPL routing for each dataset. The largest probability are in bold, and the second-largest are underlined.

Routing heuristic effectively assigns tokens to the corresponding experts. Table 2 demonstrates that PPL routing generally achieves the desired routing patterns, effectively directing inputs from a specific domain to the specialized expert models, except in the case of the MBPP dataset. Since our heuristics rely solely on inference inputs without fine-tuning, they can be considered reliable strategies. We also visualize the routing probability for both PPL and task vector routing heuristics for each dataset in Figure 9 in Appendix D. We find that PPL routing consistently produces better results than the task vector routing.

Next, we evaluate the performance on each dataset with different combinations of merging methods and routing heuristics, compared to the baseline methods. We prepare three dense finetuning baselines: **Dare Dense**, **Ties Dense** and **Random Routing** (details in Section 4.3). We also evaluate the ablation methods: merging attention layers without separation and task vector routing. We present the results of each method across datasets in Table 3. The details of training cost for each method are presented in Table 7 in Appendix.

<span id="page-7-0"></span>

| Merging            | Routing       | MBPP | HumanEval | MATH | GSM8K | NQ   | TriviaQA | Avg. |
|--------------------|---------------|------|-----------|------|-------|------|----------|------|
|                    | Dense Merging |      |           |      |       |      |          |      |
| Dare               | N/A           | 6.20 | 6.70      | 2.22 | 2.27  | 4.80 | 20.45    | 7.11 |
| Ties               | N/A           | 6.00 | 6.70      | 2.48 | 2.19  | 3.62 | 20.86    | 6.98 |
| MoE Merging        |               |      |           |      |       |      |          |      |
| Merge attention    | random        | 4.00 | 6.10      | 2.78 | 2.05  | 4.86 | 21.75    | 6.92 |
| Merge attention    | task vector   | 6.60 | 4.87      | 3.06 | 1.44  | 6.05 | 21.39    | 7.24 |
| Merge attention    | PPL           | 6.40 | 4.87      | 2.86 | 1.13  | 5.93 | 22.71    | 7.32 |
| Separate attention | task vector   | 4.00 | 7.32      | 2.98 | 2.5   | 5.37 | 20.11    | 7.05 |
| Separate attention | PPL           | 6.80 | 7.92      | 2.88 | 2.95  | 4.74 | 23.21    | 8.08 |

Table 3: Performance of proposed merging and routing methods for MoE without substantial fine-tuning and other baselines across six datasets. Separating attention layers and perplexity routing heuristics get the best average performance.

Proposed MoE method without fine-tuning outperforms the dense merging baseline. Table 3, we observe that using the PPL routing heuristic and separating attention layers achieves the best average results among all baseline methods. Compared to Random Routing and the SoTA dense merging method (Dare), our best method -PPL routing + separating attention layers - yields relative improvements of 16.8% and 13.6%, respectively. The superior performance of PPL routing aligns with Figure 9 in Appendix D, where PPL routing more accurately directs input to the appropriate experts. Moreover, the better results of separating attention layers support our expectation that this approach resolves the inconsistency of task vector counts, as discussed in Section 3.1.3.

#### 5.2 Heterogeneous Model Merging

MoE merged with heterogeneous models outperforms the corresponding experts. After showing the superiority of our homogeneous model merging method, our next question is whether the proposed heterogeneous expert merging is also ef-

<span id="page-7-2"></span>![](_page_7_Figure_7.jpeg)

![](_page_7_Figure_8.jpeg)

Figure 6: Routing probability of experts on GSM8K and MATH for the MoE w/ Olmo and MoE w/ TinyLlama.

fective. We present the performance of the dense, MoE and baseline methods in Table 4. The details of training cost for each method are presented in Table 8 in Appendix.

<span id="page-7-1"></span>

| Method                       | MBPP  | HumanEval | MATH | GSM8K | NQ   | TriviaQA | Avg.  |  |
|------------------------------|-------|-----------|------|-------|------|----------|-------|--|
| Dense Model                  |       |           |      |       |      |          |       |  |
| Base-1B                      | 4.60  | 3.04      | 2.42 | 1.44  | 6.61 | 26.72    | 7.47  |  |
| Base TinyLlama               | 5.40  | 5.27      | 2.26 | 2.2   | 8.53 | 34.27    | 9.66  |  |
| Base Olmo                    | 2.80  | 2.64      | 2.46 | 2.42  | 6.16 | 29.21    | 7.62  |  |
| Code Expert                  | 10.20 | 8.53      | 2.42 | 2.57  | 3.11 | 16.7     | 7.26  |  |
| Math TinyLlama               | 15.60 | 9.76      | 4.18 | 5.91  | 6.05 | 21.12    | 10.44 |  |
| Math Olmo                    | 0.00  | 0.00      | 4.82 | 5.08  | 3.61 | 11.25    | 4.13  |  |
| Knowledge Expert             | 3.60  | 4.26      | 2.62 | 2.04  | 5.65 | 28.71    | 7.81  |  |
| Homogeneous Expert Merging   |       |           |      |       |      |          |       |  |
| 3-expert MoE                 | 9.14  | 10.8      | 4.42 | 5.16  | 6.95 | 26.78    | 10.54 |  |
| (same data)                  | 7.14  | 10.0      | 7.72 | 5.10  | 0.75 | 20.70    | 10.54 |  |
| 3-expert MoE                 | 12.00 | 9.76      | 2.38 | 1.74  | 6.22 | 33.20    | 10.88 |  |
| (w/o math)                   | 12.00 | 9.70      | 2.36 | 1.74  | 0.22 | 33.20    | 10.00 |  |
| Heterogeneous Expert merging |       |           |      |       |      |          |       |  |
| (Ours) MoE w/                | 13.60 | 10.98     | 4.86 | 6.14  | 5.43 | 26.01    | 11.17 |  |
| Math Olmo                    | 15.00 | 10.70     |      | 0.17  | 5.75 | 20.01    | ,     |  |
| (Ours) MoE w/                | 15.80 | 11.59     | 5.42 | 6.29  | 8.25 | 32.71    | 13.34 |  |
| Math TinyLlama               | 12.00 | 11.07     |      | 0.27  | 0.20 | 52.71    | 20.04 |  |

Table 4: **Performance of proposed heterogeneous merged MoE and other baselines.** The merged MoE is comparable or outperform the dense or 3-expert baselines on the benchmark from the corresponding domain.

Table 4 shows that our merged MoE models are comparable to or outperform the domain expert models in their respective domains. For instance, the MoE merged with Math Olmo and Math TinyLlama achieves 6.14% and 6.29% accuracy on GSM8K, compared to 5.91% and 5.08% for their dense counterparts. On average, our MoEs with Olmo and TinyLlama improves performance by 43.02% and 27.78% relative to the best dense experts, respectively. Both MoEs with heterogeneous experts also outperform the 3-expert MoE baseline, particularly in math, highlighting the effectiveness of including math experts in the pipeline.

MoE merged with heterogeneous experts show the desired routing patterns in most cases. We also perform a similar routing analysis as described in Section 5.1.1. We visualize the routing probability of two MoEs when evaluating on GSM8K and MATH datasets in Figure 6 and for other datasets, we visualize the results in Figure 8 in Appendix D.

As shown in Figures [6](#page-7-2) and [8,](#page-12-4) most tokens in the coding and knowledge datasets are routed to the corresponding experts. However, unlike homogeneous model merging where the math expert has the highest routing probability for math datasets, Math Olmo or Math TinyLlama ranks second. This discrepancy is likely due to the difference in embedding outputs between the MoE and expert models. Since the MoE's embedding layer is merged from 3 Expert models and 1 other model, its output is closer to that of the Expert models, making the router more likely to select them. Adding a load balancing loss is a possible solution to address this issue [\(Sukhbaatar et al.,](#page-10-6) [2024;](#page-10-6) [Fedus et al.,](#page-9-7) [2022\)](#page-9-7), ensuring a more uniform routing distribution. We leave this for future exploration

