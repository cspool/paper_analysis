# BTS: Harmonizing Specialized Experts into a Generalist LLM

Qizhen Zhang1,2,<sup>∗</sup> , Prajjwal Bhargava1,<sup>⋄</sup> , Chloe Bi1,<sup>⋄</sup> , Chris X. Cai1,<sup>⋄</sup> , Jakob Foerster1,2,<sup>⋄</sup> , Jeremy Fu1,<sup>⋄</sup> , Punit Singh Koura1,<sup>⋄</sup> , Ruan Silva1,<sup>⋄</sup> , Sheng Shen1,<sup>⋄</sup> , Emily Dinan1,† , Suchin Gururangan1,† , Mike Lewis1,†

We present Branch-Train-Stitch (BTS), an efficient and flexible training algorithm for combining independently trained large language model (LLM) experts into a single, capable generalist model. Following [Li et al.](#page-15-0) [\(2022\)](#page-15-0), we start with a single "seed" language model which is branched into domain-specific (e.g., coding or math) experts with continual pretraining. BTS combines experts into a generalist model using lightweight stitch layers, which are inserted between frozen experts and the seed LLM, and trained on a small datamix of the expert domains. Stitch layers enable the seed LLM to integrate representations from any number of experts during the forward pass, allowing it to generalize to new domains, despite remaining frozen. Because BTS does not alter the constituent LLMs, BTS provides a modular and flexible approach: experts can be easily removed and new experts can be added with only a small amount of training. Compared to alternative model merging approaches, BTS yields the best generalist performance on a variety of downstream tasks, retaining the specialized capabilities of each of the experts.

Date: February 4, 2025

Correspondence: Qizhen Zhang at [qizhen.zhang@eng.ox.ac.uk](mailto:qizhen.zhang@eng.ox.ac.uk)

![](_page_0_Picture_8.jpeg)

# 1 Introduction

To achieve strong performance across diverse domains, large language models (LLMs) are often densely trained on trillions of tokens using thousands of GPUs [\(Dubey et al.,](#page-14-0) [2024\)](#page-14-0). Dense training requires substantial resources and significant infrastructure challenges, often requiring massive synchronization across distant compute clusters. Dense training also poses difficult datamix tradeoffs [\(Xie et al.,](#page-16-0) [2023;](#page-16-0) [Ye et al.,](#page-16-1) [2024\)](#page-16-1); for example, it can be challenging to improve performance on a new domain without forgetting the original data [\(McCloskey and Cohen,](#page-15-1) [1989;](#page-15-1) [Aghajanyan et al.,](#page-14-1) [2021\)](#page-14-1) or debug and correct unwanted behaviors without impacting others [\(Tuan et al.,](#page-15-2) [2024\)](#page-15-2).

Expert merging techniques like Branch-Train-Merge (BTM; [Li et al.,](#page-15-0) [2022;](#page-15-0) [Gururangan et al.,](#page-14-2) [2023\)](#page-14-2) address these challenges by asynchronously training distinct expert models, specialized to different domains, and merging them back into a single generalist language model by ensembling them at inference time. Experts can be removed from the mix or added as needed. However, BTM is limited because there are no learned connections between expert layers; this restricts the model's overall expressivity, especially in distant test domains. On the other hand, approaches like Branch-Train-MiX (BTX; [Sukhbaatar et al.,](#page-15-3) [2024\)](#page-15-3), which upcycles experts into an Mixture-of-Experts (MoE) model [\(Shazeer et al.,](#page-15-4) [2017\)](#page-15-4), show strong downstream task performance, but lose the flexibility and interpretability inherent in a modular approach where experts remain distinct and intact.

We present Branch-Train-Stitch (BTS), a new algorithm for building a generalist LLM from a collection of smaller expert models which achieves the best generalist model performance. Like other merging techniques [\(Li et al.,](#page-15-0) [2022;](#page-15-0) [Gururangan et al.,](#page-14-2) [2023\)](#page-14-2), BTS begins with a training phase in which experts are created via independent continued pretraining on domains of interest (starting from a shared "seed" checkpoint; [Li et al.](#page-15-0) [2022\)](#page-15-0). After expert training, the experts are adapted into a unified, generalist model by inserting and training stitch layers between models, while keeping the experts themselves frozen.

<sup>1</sup>GenAI at Meta, <sup>2</sup>University of Oxford

<sup>∗</sup>First author, work done at Meta, <sup>⋄</sup>Ordered alphabetically, Joint last author

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1 Overview of the BTS algorithm. BTS operates in three phases. Different colors correspond to different expert domains. 1) Branch: Following [Li et al.](#page-15-0) [\(2022\)](#page-15-0), we begin with a pretrained seed model and create N copies of it. 2) Train Experts: Each copy is independently pretrained on its respective data mixture, resulting in specialized expert models, as described in [Li et al.](#page-15-0) [\(2022\)](#page-15-0). 2) Stitching: Stitch layers are inserted throughout the layers, alternating between the Experts-into-Hub stitch layer and the Hub-into-Experts stitch layer. Only the stitch layers are updated during this training phase. The BTS model always have a Experts-into-Hub stitch layer as the last layer, as the hub output is returned as the final BTS output.

This stitching architecture adds connections between experts via a gating mechanism on top of the language model layer outputs which determine how hidden states from one expert flow into another. One can imagine several ways to combine representations produced by different experts: all experts can directly connect to all other experts, only certain experts can connect to certain others, and everything in between. We opt for a hub-and-spoke model, in which a central "hub" model (the seed LLM) can update its own representations via the spokes (specialized experts), and vice versa, but the experts have no direct connection to each other. This design choice balances efficiency and performance. Since the seed model is trained on a variety of data, it is a natural choice for the hub, so all of our experiments adopt this set-up. For each layer in the forward pass, the stitching architecture alternates between hub-to-expert merging, where the hidden representations of the experts are updated with a projected hub LLM representation, and hub-to-expert merging, where the hub's hidden representation is updated with a combined hidden representation of all experts. The final output provided by the merged LLM is the output of the seed model. These design choices are further motivated and validated empirically with ablations in [Section 4.](#page-8-0)

In experiments [\(Section 3\)](#page-4-0), we find that BTS achieves the best generalist model performance compared to both expert merging and expert upcycling baselines and can even perform better than some individual experts on their target tasks. Notably, this is achieved with training only the small set of stitching parameters. The modular design of BTS, in which individual experts remain unchanged in the merging process, offers flexibility and interpretability. Targeted performance improvements for specific domains can be achieved completely asynchronously. Furthermore, downstream behaviors can be easily understood by analyzing which experts are 'active' at any given token, providing transparency into the model's decision-making process.

Our contributions are summarized as follows:

- Branch-Train-Stitch, Section 2: We propose Branch-Train-Stitch, an efficient and flexible approach for stitching distinct expert models into a more powerful, generalist LLM.
- Experiments, Section 3: We validate this approach through experiments on seed language models of 2.7B parameters. Our results demonstrate that BTS outperforms competitive baselines in downstream task performance, achieving the best average performance across benchmarks.
- Analysis, Section 4: We motivate the BTS architectural choices with ablations and investigate the impact on "cross capability" tasks, i.e. tasks at the intersection of expert domains, and show that, in certain settings, BTS can achieve cross capability performance greater than any expert. Finally, we provide detailed analysis of the behavior of stitch layers at inference time, showing that BTS can dynamically adjust its expert utilization even within the same prompt.

## <span id="page-2-0"></span>2 Branch-Train-Stitch

This section provides an overview of the BTS algorithm, beginning with a brief background on language model architectures (Section 2.1), followed by a detailed description of the BTS methodology (Section 2.2) and architecture (Section 2.3).

#### <span id="page-2-1"></span>2.1 Language model architecture background

Transformer The typical architecture of large language models (LLMs) is built by stacking multiple Transformer blocks (Vaswani et al., 2017). Each Transformer block consists of a Multi-Headed Attention module, commonly referred to as the *attention layer*, followed by a residual connection and a feed-forward neural network (FFN).

Mixture-of-Experts The Mixture of Experts (MoE; Shazeer et al., 2017) model replaces the FFN in the Transformer by an MoE layer. An MoE layer consists of a linear router and a set of N FFN experts, denoted as  $\{FFN_i(x)\}_{i=1}^N$ . The router produces normalized router logits p(x) for the input representation x, where  $p_i(x)$  is the gating value for the i-th FFN expert, FFN $_i$ . The router assigns the input representation x to a subset of experts,  $\mathcal{T}$ , with the highest gating values. The final output of the MoE layer is the weighted sum of the selected experts' outputs, weighted by their gating values:

<span id="page-2-3"></span>
$$y_{\text{MoE}} = \sum_{i \in \mathcal{T}} p_i(x) \text{FFN}_i(x).$$
 (1)

Mixture-of-Attention Mixture of Attention (MoA; Zhang et al., 2022) extends MoE by also replacing the attention layer in Transformers with an MoA layer. Similar to the MoE layer, an MoA layer comprises of a set of N attention experts (denoted as  $\{Attention_j(x)\}_{j=1}^N$ ), a linear router that outputs normalized router logits q(x). Like the MoE, the MoA layer's final output is a gating-value weighted sum of the computations from the selected attention experts  $\mathcal{M}$ :

<span id="page-2-4"></span>
$$y_{\text{MoA}} = \sum_{i \in \mathcal{M}} q_i(x) \text{Attention}_i(x).$$
 (2)

#### <span id="page-2-2"></span>2.2 BTS algorithm overview

The BTS algorithm involves three stages, resulting in an efficiently-trained generalist dense model. The process is visualized in Figure 1.

1. **Branch**: Following Li et al. 2022, given a pretrained Transformer seed model  $m_0$ , we create n copies of the model  $m_1, ..., m_n$ .

- 2. **Train**: Also following Li et al. 2022, each copy of the seed model  $m_i$  independently undergoes a continued pretraining phase on a specialized data mixture,  $\mathcal{D}_i$ , each tailored to different domains such as code, mathematics, and multilingual (Gururangan et al., 2020). This phase yields specialized models that have enhanced performance within their respective domains compared to the seed model  $m_0$ . However, these models might perform worse in domains outside of their specialization as they forget knowledge from the initial pretraining phase. We refer to these models  $m_i$  as experts, and note that this usage of the term "expert" differs in meaning from the FFN / attention experts in MoE and MoA models.
- 3. **Stitch**: We merge the seed  $(m_0)$  and expert models  $(m_i, i > 0)$  from the previous steps using our lightweight stitch layers  $\Psi$ , which are trained for a small number of steps on a mixture of data from expert domains. The stitch layer architecture is described in Section 2.3. Importantly, *only* the stitch layers are updated during this phase, while the parameters of the seed and expert models remain frozen. This ensures that BTS training is a flexible approach experts can be added or removed after merging, only requiring retraining stitch layer parameters.

#### <span id="page-3-0"></span>2.3 Model architecture

Next, we provide additional details on the BTS architecture (Figure 1). We introduce the *stitch layer*, which, as mentioned above, merges n+1 Transformer models  $m_0, ..., m_n$ . We designate  $m_0$  as the *hub* and  $m_1, ..., m_n$  as the *experts*. The hub is usually the seed model, unless otherwise noted.

Suppose the expert  $m_i$  contains L Transformer layers,  $\{\ell_i^j\}_{j=1}^L$ . We insert K stitch layers – one each after every  $\lfloor \frac{L}{K} \rfloor$  Transformer layers. We denote  $\Psi_j$  as the stitch layer inserted after Transformer layers  $\{\ell_i^j\}_{i=0}^n$ . The stitch layer  $\Psi_j$ , takes as input the hidden states, or outputs, from the hub's j-th layer  $\ell_0^j$  and the experts' j-th layers,  $\{\ell_i^j\}_{i=1}^n$ . We denote the hidden states respectively as  $h_0^j$  for the hub and  $\{h_i^j\}_{i=1}^n$  for the experts. The outputs of the stitch layer,  $\Psi_j(h_0^j, \ldots, h_n^j) = (\tilde{h}_0^j, \ldots, \tilde{h}_n^j)$ , become the input to the corresponding experts  $m_i$ 's j + 1-th layer  $(\ell_i^{j+1})$ .

Each stitch layer  $\Psi$  introduces two sets of learnable parameters:

- 1. Linear projections,  $\{w_{\text{proj}_1},...,w_{\text{proj}_n}\}$ , where  $w_{\text{proj}_i} \in \mathbb{R}^{\dim \times \dim}$  either projects the expert hidden states to the hub model's hidden state space or projects the hub model's hidden state into the expert's hidden state space.
- 2. A linear gate  $w_{\text{gate}} \in \mathbb{R}^{\dim \times \dim \times n}$ , which computes the contribution of each model's hidden state.

To apply these gates, we alternate between two types of stitch layers (refer to Figure 1 for the illustration and Appendix B for the pseudo code):

The Experts-into-Hub Stitch Layer In this layer, the expert models' hidden states are first projected into the hub model's hidden state space,. The hub then combines its own representation with the projected experts' hidden states, weighted by the outputs of a softmax-based gating mechanism.

$$g = \operatorname{softmax}(\operatorname{dropout}(w_{\operatorname{gate}}(h_0)))$$

$$\tilde{h}_i = w_{\operatorname{proj}_i}(h_i) \qquad \text{for } i \in \{1, ..., n\}$$

$$\tilde{h}_0 = h_0 * g_0 + \sum_{i=1}^n g_i * \tilde{h}_i,$$

$$(3)$$

where  $q_i$  correspond to the *i*-th expert in the gate value q.

The Hub-into-Experts Stitch Layer In this layer, the hub representation is projected into each of the expert model's hidden state space. Each expert combines its own hidden state with a gated projection of the hub representation using a sigmoid-based gating mechanism:

$$g = \operatorname{Sigmoid}(\operatorname{dropout}(w_{\operatorname{gate}}(h_0)))$$

$$\tilde{h}_0 = h_0$$

$$\tilde{h}_i = (1 - g_i) * h_i + g_i * w_{\operatorname{proj}_i}(h_0)$$
for  $i \in \{1, ..., n\}$ 

<span id="page-4-0"></span>As we demonstrate in [Section 4,](#page-8-0) this alternating architecture is essential for enabling cross capabilities without degrading generalist performance.

# 3 Results: Building a generalist model

We validate the BTS approach through experiments with a seed language models of 2.7B parameters. We describe model [\(Section 3.1\)](#page-4-1), data [\(Section 3.2\)](#page-4-2), baseline [\(Section 3.3\)](#page-5-0), and evaluation [\(Section 3.4\)](#page-6-0) details and discuss experimental results in [Section 3.5.](#page-7-0)

## <span id="page-4-1"></span>3.1 Model details

<span id="page-4-3"></span>Seed model We pretrain a 2.7B parameter language model, following the same text recipe used in Llama 3 [\(Dubey et al.,](#page-14-0) [2024\)](#page-14-0). See [Table 1](#page-4-3) for architecture details. We employ a learning rate schedule that warms up from 0 to 4e-4 over 2000 steps, then undergoes a cosine decay to 1% of the peak learning rate. The seed model is trained for 2.2M steps on 15T tokens.

| Layers                | 20                  |
|-----------------------|---------------------|
| Model Dimension       | 3072                |
| FFN Dimension         | 12288               |
| Attention Heads       | 24                  |
| Key/Value Heads       | 1                   |
| Activation Function   | SwiGLU              |
| Vocabulary Size       | 128,000             |
| Positional Embeddings | RoPE (θ = 500, 000) |
|                       |                     |

Table 1 Architecture details for the 2.7B parameter seed model and expert models.

Expert models We create three copies of the seed model, each of which is continually trained for 96k training steps over a 200B token specialized data mixture to produce expert models for code, mathematics, and multilingual tasks. During the continued pretraining phase, we use a batch size of 2M tokens and a learning rate of 5e-6, followed immediately by a cosine decay schedule that reduces the learning rate to 1% of its initial value. This learning rate is derived by annealing from the final learning rate used at the end of seed model pretraining, adjusted to account for the reduced batch size in this continued pretraining phase. We adopted this learning rate strategy as it yielded the most stable learning during the continued pretraining phase

<span id="page-4-2"></span>BTS model We use four stitch layers to combine the seed model together with the three expert models. The four stitch layers are inserted after every five layers in the seed and expert models. We refer to the resulting model as the BTS model. As described in [Section 2,](#page-2-0) the four stitch layers alternate between a Merge-into-Expert layer and Experts-into-Hub stitch layer. Upon initialization, the BTS model is further trained for 15B tokens over 7000 steps using a batch size of 2M tokens. The optimization objective is to minimize the next-token prediction loss from the hub model's output. The learning rate schedule warms up from 0 to 5e-6 over 2000 steps, then undergoes a cosine decay to 1% of the peak learning rate. Note that during the BTS training phase, only the stitch layers are updated while all the parameters of the seed model and the expert models are frozen.

<span id="page-5-1"></span>

|                  | Training Params | Total Params | Active Params |
|------------------|-----------------|--------------|---------------|
| Expert upcycling |                 |              |               |
| BTX Sample       | 7.2B            | 7.2B         | 2.9B          |
| BTX Soft         | 7.2B            | 7.2B         | 7.2B          |
| BAM              | 8.4B            | 8.4B         | 8.4B          |
| Expert merging   |                 |              |               |
| Model Soup       | N/A             | 2.7B         | 2.7B          |
| BTM              | N/A             | 10.8B        | 10.8B         |
| Expert Routing   | 15k             | 10.8B        | 2.7B          |
| BAM Adapters     | 1.5B            | 9.9B         | 9.9B          |
| BTS              | 264M            | 11B          | 11B           |

Table 2 Training, total, and active parameter count for BTS and baselines. We use "expert upcycling" to describe MoE upcycling methods where the seed and experts themselves do not remain intact during the MoE training phase. These methods require significantly more training parameters, and thus are less modular, less flexible, and less interpretable. We use "expert merging" to describe methods, such as BTS, where the seed and expert models remain frozen during the merging phase. Expert merging methods require minimal number of training parameters, making them more modular and interpretable.

## 3.2 Data details

Seed model We adopt the same text pretraining mixture as Llama 3 [\(Dubey et al.,](#page-14-0) [2024\)](#page-14-0).

Expert models In the continued pretraining phase, each dense expert is trained on a specialized data mixture for 200B tokens:

- Code: We adopt a recipe similar to that of CodeLlama [\(Rozière et al.,](#page-15-6) [2023\)](#page-15-6) with > 85% code tokens, utilizing the code data subset of the seed model mixture.
- Math: We continue pretraining on the OpenWebMath dataset [\(Paster et al.,](#page-15-7) [2023\)](#page-15-7).
- Multilingual: We utilize a mixture of 90% non-English data and 10% English data, with each subset pulled from the seed model mixture, following the multilingual expert recipe described in [Dubey et al.](#page-14-0) [\(2024\)](#page-14-0).

BTS model The data mixture for the BTS training phrase consists of 15% expert domain tokens for each of the code, math, and multilingual domains. The remaining 55% of the mixture consists of the pretraining data utilized for the seed model outside of these domains.

## <span id="page-5-0"></span>3.3 Baselines

In addition to the seed and expert models, we also compare BTS with expert upcycling and expert merging baselines. We use expert upcycling to describe methods where the seed and expert models are used to initialize an MoE model, which is further trained. The entire MoE is updated during training and as such the experts and seed model themselves do not remain intact. This approach loses the flexibility and interpretability inherent in a more modular approach, and any model change requires updating a large number of parameters. On the other hand, we use expert merging to describe methods, such as BTS, in which the seed and expert parameters remain frozen during the merging phase.

#### Expert upcycling baselines:

- BTX [\(Sukhbaatar et al.,](#page-15-3) [2024\)](#page-15-3): We upcycle the seed model and three expert models into an MoE. Our baselines include two BTX variants, where the FFN experts employ one of two routing strategies: 1) sample top-1 routing [\(Sukhbaatar et al.,](#page-15-3) [2024\)](#page-15-3), where we use a Gumbel-Softmax [\(Jang et al.,](#page-14-4) [2016\)](#page-14-4) for the routing function, and 2) soft-routing, where all four experts are activated at all times. We use the same experimental setup as BTS runs, including training data and the learning rate schedule. See [Section 2.1](#page-2-1) for details on the MoE architecture.
- BAM [\(Zhang et al.,](#page-16-3) [2024\)](#page-16-3): We upcycle the seed model and the three expert models into an MoE with both attention experts and FFN experts. See [Section 2.1](#page-2-1) for a description of the attention experts architecture. We employ soft-routing for both sets of experts, ensuring that, like BTS, all FFN and attention parameters of the seed and expert models are activated during training and inference. We use the same experiment setup as BTS runs.

#### Expert merging baselines:

- Model soup [\(Wortsman et al.,](#page-15-8) [2022\)](#page-15-8): We uniformly average the weights of the seed and expert models. Unlike other baselines, no further training is required upon initialization.
- BTM [\(Li et al.,](#page-15-0) [2022\)](#page-15-0): We ensemble the output logits of the seed and expert models. The ensemble weights are estimated using Bayes' rule with a uniform prior [\(Li et al.,](#page-15-0) [2022;](#page-15-0) [Gururangan et al.,](#page-14-2) [2023\)](#page-14-2). Like the model soup baseline, no further training is required upon initialization.
- Expert routing: We train a linear router ∈ R dim×<sup>n</sup> that routes to either the seed model or one of the expert models. The router's training objective is a classification cross-entropy loss where the target is the model with the smallest next-token prediction loss for the input. Given a prompt, the router decides on the model and routes all subsequent tokens to the same model. During training, the routing decision is made based on the average embedding of the first t tokens in the input, where t is randomly sampled between 32 and 256. During inference, the routing decision is made based on the average embedding of the entire prompt. We train the linear router with a constant learning rate of 5e-4 and batch size of 1M. The model is trained for 1B tokens only, as we did not see an improvement in downstream metrics or training loss with further training.
- BAM with adapters [\(Zhang et al.,](#page-16-3) [2024\)](#page-16-3): We train an expert-intact variant of BAM with soft-routing, which we refer to as BAM with adapters. In this variant, each attention expert and each FFN expert's output undergo a linear adapter layer Wproj<sup>i</sup> ∈ R dim×dim. Formally, we replace [Equation 1](#page-2-3) and [Equation 2](#page-2-4) by the following:

$$y_{\text{MoE}} = \sum_{i \in \mathcal{T}} p_i(x) W_{\text{ffn proj}_i} (\text{FFN}_i(x))$$

$$y_{\text{MoA}} = \sum_{i \in \mathcal{M}} q_i(x) W_{\text{attn proj}_i} (\text{Attention}_i(x)).$$
(5)

Only the router and adapters are updated during training, while all other parameters remain frozen. We use the same experiment setup as BTS runs.

We show a comparison of the number of training, active, and total parameters in [Table 2.](#page-5-1) Note that BTS has the most total parameters of all variants, but only a small fraction of the training parameters of the expert upcycling variants.

## <span id="page-6-0"></span>3.4 Evaluation

We assess model performance with zero-shot and few-shot downstream tasks relevant to the expert domains.

- General Knowledge and Reasoning: To assess general knowledge and reasoning capabilities, we report MMLU (5-shot; [Hendrycks et al.,](#page-14-5) [2021a\)](#page-14-5) and Big-Bench Hard (3-shot; [Suzgun et al.,](#page-15-9) [2022\)](#page-15-9). In tables, we denote Big-Bench Hard as BBH.
- Code: For code generation capabilities, we evaluate on MBPP (3-shot; [Austin et al.,](#page-14-6) [2021\)](#page-14-6) and HumanEval (0-shot; [Chen et al.,](#page-14-7) [2021\)](#page-14-7) benchmarks. We denote HumanEval as HE in the results table for brevity.

<span id="page-7-1"></span>

|                   |       | General |       | Code  | Multilingual |           | Math  |           |       |
|-------------------|-------|---------|-------|-------|--------------|-----------|-------|-----------|-------|
|                   | MMLU  | BBH     | MBPP  | HE    | Flores(S)    | Flores(T) | GSM8K | MATH      | Avg.  |
| 2.7B Dense models |       |         |       |       |              |           |       |           |       |
| Seed Model        | 28.4  | 35.6    | 27.0  | 20.7  | 29.5         | 35.7      | 10.5  | 4.82      | 24.0  |
| Code Expert       | 30.3  | 35.2    | 32.0  | ∗25.0 | 29.0         | 35.5      | 11.4  | 4.40      | 25.4  |
| Multiling. Expert | 26.6  | 34.7    | 26.2  | 18.3  | ∗31.9        | ∗37.1     | 10.8  | 4.16      | 23.7  |
| Math Expert       | ∗36.3 | ∗37.2   | 26.2  | 16.5  | 23.6         | 32.7      | ∗20.5 | 10.1      | 25.4  |
| Expert upcycling  |       |         |       |       |              |           |       |           |       |
| BTX Sample        | 30.4  | 36.6    | 30.0  | 21.3  | 30.5         | 36.0      | 13.9  | 6.58      | 25.7  |
| BTX Soft          | 34.7  | 36.8    | 29.6  | 23.2  | 31.0         | 36.0      | 19.2  | 9.10      | 27.4  |
| BAM               | 35.2  | 37.1    | 29.8  | 22.6  | 31.0         | 36.1      | 20.3  | 10.1      | 27.8  |
| Expert merging    |       |         |       |       |              |           |       |           |       |
| Model Soup        | 30.7  | 37.0    | 29.6  | 22.6  | 29.5         | 36.2      | 13.6  | 6.46      | 25.7  |
| BTM               | 30.6  | 37.0    | 31.8  | 23.8  | 31.8         | 37.0      | 12.7  | 10.1      | 26.9  |
| Expert Routing    | 28.4  | 35.6    | 27.0  | 23.8  | 30.8         | 37.0      | 10.5  | 5.04      | 24.8  |
| BAM Adapters      | 34.0  | 37.0    | 28.8  | 22.6  | 31.0         | 36.1      | 18.8  | 10.0      | 27.3  |
| BTS               | 35.8  | 36.9    | ∗32.2 | 22.0  | 30.9         | 36.2      | 20.2  | ∗<br>10.6 | ∗28.1 |

Table 3 Performance of BTS against expert merging and upcycling methods, seed and expert models measured on popular benchmarks across several capabilities. Bolded numbers indicate the best performance among dense models or merged models, while an asterisk (<sup>∗</sup> ) denotes the best performance across all models. See [Section 3.4](#page-6-0) for benchmark details. Although dense expert models sometimes achieve the best results in their specialized domains, they often significantly under-perform in other domains. Among all merged models, BTS achieves the best average performance. Notably, BTS not only emerges as the most well-rounded generalist expert but also outperforms the corresponding domain-specific experts on MATH and MBPP tasks.

- Multilingual: For measuring multilingual capabilities, we use machine translation sub-tasks in Flores (1-shot; [Goyal et al.,](#page-14-8) [2022\)](#page-14-8). Specifically, we evaluate on seven languages: Dutch, Spanish, Portuguese, Vietnamese, Indonesian, Hindi, and French. We display the sub-tasks evaluations into two categories, 1) those with English as the source translation language (S), and 2) those with English as the target translation language (T).
- Math: For mathematical reasoning, we report the performance on GSM8K (8-shot; [Cobbe et al.,](#page-14-9) [2021\)](#page-14-9) and MATH (4-shot; [Hendrycks et al.,](#page-14-10) [2021b\)](#page-14-10).

## <span id="page-7-0"></span>3.5 Results

Results on general knowledge, code, multilingual, and math benchmarks for the seed model, expert models, and all expert merging and expert upcycling baselines are reported in [Table 3.](#page-7-1) We make the following observations:

- Expertmodels highlight datamix tradeoffs: While the dense expert models typically achieve the best results in their respective target domains, they often significantly underperform in other domains, highlighting that improving performance in one domain may come at the cost of regressing in others. For example, the Math expert outperforms all models in GSM8K, but lags behind the seed model substantially in coding tasks.
- Learned connections are important for expressive merging: Methods like BAM with adapters and BTS outperform expert merging methods without learned connections between experts, such Model Soup, BTM, and Expert Routing. This demonstrates the importance of adding learned, intermediate connections between experts.
- BTS achieves the best generalist performance: Among all model variants seed, expert, expert merging, and experts upcycling – BTS achieves the best average performance across tasks. Notably, BTS achieves

similar or better performance to the expert upcycling baselines at only a fraction of the training parameters.

• BTS can outperform individual experts in their specialized tasks: BTS emerges not only as the most well-rounded generalist model, but is also the only model which achieves better performance than any individual expert in some tasks. BTS outperforms the Code expert in MBPP and the Math expert in the MATH task.

# <span id="page-8-0"></span>4 Ablations and analysis

We provide detailed ablations and analysis as follows:

- Enabling cross capabilities, [Section 4.1:](#page-8-1) We evaluate how BTS performs on cross capabilities, or capabilities at the intersection of two or more expert specialties and compare to other merging techniques.
- BTS architecture design, [Section 4.2:](#page-9-0) We empirically validate several BTS architecture choices, including assessing the impact of the number of stitch layers, the alternating stitch layer design, and choice of hub model.
- Interpreting the BTS stitch layers, [Section 4.3:](#page-10-0) Finally, we provide visualizations and analysis of how the BTS stitch layer gate values behave at inference time for various downstream tasks.

## <span id="page-8-1"></span>4.1 Enabling cross capabilities

In addition to evaluating merged models on the union of the expert capabilities, we also explore whether merged models can demonstrate entirely new capabilities at the intersection of expert specialties [\(Zhong](#page-16-4) [et al.,](#page-16-4) [2024\)](#page-16-4). For example – can a Russian-language expert and a Math expert be combined in such a way that the merged model performs better than either expert at Russian math tasks? We refer to these as cross capabilities.

#### 4.1.1 Cross capabilities experimental set-up

In order to evaluate cross capabilities, we train an additional Russian-language expert specifically on Russian data, and all merged models are created with only the Russian and Math experts. We make these choices in order to study cross capability emergence in a controlled setting:

- Reducing cross capability expert contamination: We found that our coding data contained significant portions of non-English natural language, affecting the Code expert's ability in multilingual reasoning tasks, so we remove this model from this mix [\(Blevins and Zettlemoyer,](#page-14-11) [2022\)](#page-14-11). We further remove the seed model which contains both multilingual and math data.
- Prevalance of cross capability training and evaluation data: We limit our study to languages in which we have cross capability data to both train and evaluate the models on — for this reason, we focused on Russian and Math.

Note that when merging only two experts, there is no notion of "hub" model: the stitch layers alternate between merging Russian-into-Math and Math-into-Russian.

During the expert merging or expert upcycling training phase, we train on 2B tokens of Russian mathematics data extracted from web data using a combination of language identification (LID) and math classifiers. We found this additional cross capability in-domain training data was essential. Without it, all merged models struggle to achieve good cross capability performance (see experiments in [Appendix A\)](#page-17-0).

We introduce an additional baseline via continued pretraining the strongest dense model, the seed model, in a data-matched manner on the Russian mathematics data. This is to evaluate the impact of training on in-domain data without increasing the overall model capacity. Additional details of the experimental set-up are provided in [Appendix A.](#page-17-0) All models are evaluated on the Russian subset of MGSM (8-shot; [Shi et al.](#page-15-10) [2022\)](#page-15-10), which are Russian translations of examples from GSM8K [\(Cobbe et al.,](#page-14-9) [2021\)](#page-14-9).

<span id="page-9-1"></span>

|                  | Flores |       |       |           |  |  |  |  |
|------------------|--------|-------|-------|-----------|--|--|--|--|
|                  | GSM8K  | En/Ru | Ru/En | Ru-MGSM   |  |  |  |  |
| Dense models     |        |       |       |           |  |  |  |  |
| Seed Model       | 10.5   | 22.8  | 32.8  | 12.8      |  |  |  |  |
| Math Expert      | ∗20.5  | 10.2  | 28.9  | 10.8      |  |  |  |  |
| Russian Expert   | 9.48   | ∗32.3 | 34.6  | 9.60      |  |  |  |  |
| Seed Model (DM)  | 12.6   | 24.8  | 32.8  | 14.0      |  |  |  |  |
| Expert upcycling |        |       |       |           |  |  |  |  |
| BTX Sample       | 15.6   | 29.9  | 34.3  | 17.6      |  |  |  |  |
| BTX Soft         | 17.6   | 30.6  | 34.5  | 17.6      |  |  |  |  |
| BAM              | 19.3   | 30.9  | 34.5  | ∗<br>18.4 |  |  |  |  |
| Expert merging   |        |       |       |           |  |  |  |  |
| Model Soup       | 17.5   | 14.7  | 32.3  | 13.2      |  |  |  |  |
| BTM              | 20.5   | ∗32.3 | 34.6  | 9.60      |  |  |  |  |
| Expert Routing   | 9.48   | ∗32.3 | 34.6  | 9.60      |  |  |  |  |
| BAM Adapters     | 15.2   | 31.0  | 34.3  | 15.6      |  |  |  |  |
| BTS              | 13.3   | 31.9  | ∗34.7 | 16.0      |  |  |  |  |

Table 4 Cross capability performance. We evaluate the seed model, Russian-language, and Math experts on the Russian subset of MGSM [\(Shi et al.,](#page-15-10) [2022\)](#page-15-10). We compare their performance with expert merging and expert upcyling baselines trained with small amounts of in-domain data on Russian mathematics. We also continued pretraining the strongest dense model, the seed model on the same in domain data. We call the resulting baseline "Seed Model Data Matched (DM)". Bolded numbers indicate the best performance among dense models or merged models, while an asterisk (<sup>∗</sup> ) denotes the best performance across all models. BTS outperforms the data-matched seed model, and achieves the best cross capability performance among all expert merging methods. This demonstrates that with only a small amount of in-domain data, BTS models can effectively learn how to combine expert capabilities.

#### 4.1.2 Cross capabilities results

See [Table 4](#page-9-1) for cross capability results on Russian MGSM. Notably, we see that BTS can effectively leverage both experts to excel at a new task, surpassing the data-matched seed model baseline, even though the experts themselves remain unchanged: by adding connections between them, the resulting model exceeds the sum of its individual parts. Among all expert-merging baselines, BTS achieves the best cross capability performance. BTX and BAM variants also show strong performance, outperforming BTS, likely due to their significantly greater training capacity on in-domain data.

## <span id="page-9-0"></span>4.2 BTS architecture design

We ablate the impact of the number of stitch layers, the alternating stitch layer architecture, and the hub model selection.

#### 4.2.1 Impact of the number of stitch layers

<span id="page-9-2"></span>

|           | General |      | Code |           |           | Multilingual | Math  |      |      |
|-----------|---------|------|------|-----------|-----------|--------------|-------|------|------|
|           | MMLU    | BBH  | MBPP | HumanEval | Flores(S) | Flores(T)    | GSM8K | MATH | Avg. |
| 10 Layers | 36.1    | 37.8 | 31.8 | 22.0      | 31.2      | 36.5         | 19.1  | 10.4 | 28.1 |
| 4 Layers  | 35.8    | 36.9 | 32.2 | 22.0      | 33.9      | 36.2         | 20.2  | 10.6 | 28.1 |
| 1 Layer   | 34.9    | 37.8 | 29.6 | 19.5      | 30.8      | 35.9         | 17.7  | 9.9  | 27.0 |

Table 5 Ablations on the effect of varying number of stitch layers on downstream task performance. The first two rows are configurations with 10 and 4 stitch layers distributed uniformly throughout the seed and expert models. The third row is a configuration with a single Experts-into-Hub stitch layer placed after the last dense model layers. The 10 and 4 layers configuration performs similarly, but the single-layer configuration lags behind model performance significantly.

We measure the impact of varying the number of stitch layers on model performance, as shown in [Table 5.](#page-9-2) The first two rows present configurations with 10 and 4 stitch layers, respectively, distributed uniformly throughout the seed and expert models. In the third row, we investigate a configuration with a single Experts-into-Hub stitch layer placed after the final language model layers.

Our ablations show that a single stitch layer is insufficient for learning to effectively merge capabilities, as its performance lags significantly behind configurations with 4 or 10 layers. This also demonstrates that the BTS models with more than one stitch layer combine models in a more expressive way than than simply combining output representations. The 4 and 10 layer configurations perform similarly, however, we note that this may be due to under-training of the 10 layer variant as all models are trained on the same number of tokens.

#### 4.2.2 Importance of the alternating stitch layer architecture

The BTS architecture involves alternating between the Experts-into-Hub stitch layer and the Hub-into-Experts stitch layer. We ablate the impact of adopting this alternating architecture, as opposed to utilizing all Experts-into-Hub layers. As shown in [Table 6,](#page-10-1) the alternating architecture (first row) yields significantly better cross capability performance compared to using only homogeneous Experts-into-Hub stitch layers (second row). However, both the alternating and non-alternating architectures achieve comparable performance on generalist tasks, as shown in [Table 7.](#page-10-2) These results demonstrate that an alternating architecture is essential for achieving cross capability performance while maintaining strong generalist performance.

<span id="page-10-1"></span>

|                                              | Flores       |              |              |              |  |  |  |
|----------------------------------------------|--------------|--------------|--------------|--------------|--|--|--|
|                                              | GSM8K        | En/Ru        | Ru/En        | Ru-MGSM      |  |  |  |
| BTS Alternating<br>BTS Experts-into-Hub Only | 13.3<br>15.2 | 31.9<br>32.0 | 34.7<br>35.0 | 16.0<br>11.6 |  |  |  |

Table 6 Comparison of alternating and non-alternating BTS variants cross capabilities tasks with additional in-domain Russian math training data. The alternating variant significantly outperforms the non-alternating variant.

<span id="page-10-2"></span>

|                      | General |      | Code |      | Multilingual |           | Math  |      |      |  |
|----------------------|---------|------|------|------|--------------|-----------|-------|------|------|--|
|                      | MMLU    | BBH  | MBPP | HE   | Flores(S)    | Flores(T) | GSM8K | MATH | Avg. |  |
| BTS Alternating      | 35.8    | 36.9 | 32.2 | 22.0 | 30.9         | 36.2      | 20.2  | 10.6 | 28.1 |  |
| All Experts-into-Hub | 36.1    | 37.9 | 32.4 | 22.6 | 31.4         | 36.4      | 19.9  | 10.8 | 28.4 |  |

Table 7 Comparison of alternating and non-alternating BTS variants on generalist tasks. Both variants achieves similar performance on most domains, with the non-alternating variant slightly outperforming the alternating variant on average.

#### 4.2.3 Impact of hub model selection

By default, we always use the seed model as the hub model in BTS. This design choice is motivated from the fundamental nature of the seed model: as all experts are initialized from the seed model, the seed model's representations are more closely aligned with the experts' than the experts' are with each other, which may allow for more effective merging of representations via the BTS stitch layers.

To validate this hypothesis, we conduct an ablation study in which we use an expert model as the hub instead. Specifically, we select the Math expert for this experiment, as it has the best generalist performance among all expert models. The seed model then is used as one of the "experts" or spoke models in BTS. As shown in [Table 8,](#page-11-0) the results indicate that across most downstream tasks, selecting the seed model as the hub significantly outperforms using an expert model as the hub, validating this design choice.

## <span id="page-10-0"></span>4.3 Interpreting the BTS stitch layers

The gate values of the Experts-into-Hub stitch layer determine the weight of each expert in the combined representation. Intuitively, the higher the expert or seed model's gate values, the more important this model

<span id="page-11-0"></span>

|          | General |      | Code |      | Multilingual |           | Math  |      |      |
|----------|---------|------|------|------|--------------|-----------|-------|------|------|
|          | MMLU    | BBH  | MBPP | HE   | Flores(S)    | Flores(T) | GSM8K | MATH | Avg. |
| Seed Hub | 35.8    | 36.9 | 32.2 | 22.0 | 30.9         | 36.2      | 20.2  | 10.6 | 28.1 |
| Math Hub | 33.9    | 37.8 | 30.7 | 20.1 | 29.8         | 36.0      | 15.6  | 5.73 | 26.2 |

Table 8 Comparison of utilizing the seed model as the hub versus an expert. We ablate BTS (row 1) with a variant where we instead use the Math expert model as the hub (row 2). Using the seed model as the hub significantly outperforms using an expert model as the hub across most downstream tasks. This confirms that using the seed model as the hub in BTS is important for achieving strong generalist performance.

is for the task. We inspect these values to get insight into the model's decision-making progress on various tasks.

#### 4.3.1 Visualizing gate values on expert specialty tasks

[Figure 2](#page-11-1) visualizes how the gate values of the last stitch layer, an Experts-into-Hub stitch layer, vary when generating a sequence during inference on various expert specialty tasks. The first row plots the gate values for prompt tokens, while the second row plots the gate values for the generated tokens. Each column corresponds to a different prompt, sampled from the corresponding benchmark task.

This visualization shows that the gate values align closely with the task requirements – with the specialized expert associated with the task typically dominating the gate values – while effectively mixing representations from different models over the course of the sequence. For example, for the the math task, GSM8K, the math expert has the highest gate value over the course of the generation while the other models' gate values are nearly zero. For language translation task, Flores, the multilingual expert and the seed model dominate, with each model being relied on more heavily at different parts of the prompt or generation.

<span id="page-11-1"></span>![](_page_11_Figure_6.jpeg)

Figure 2 Visualization of how BTS gate values vary when generating a sequence during inference. We inspect the gate values for the last stitch layer over the course of a sequence. The first row plots the gate values for prompt tokens, while the second row plots the gate values for the generated tokens. Each column corresponds to a different prompt, sampled randomly from the corresponding benchmark task.

#### 4.3.2 Visualizing gate value transitions on context-switching tasks

Unlike merge methods which make sequence-level choices about which expert to use, BTS can effectively context switch over the course of the sequence, seamlessly transitioning between different tasks. [Figure 3](#page-12-0) illustrates the gate values of BTS's final stitch layer when processing context-switching prompts. These prompts are constructed by concatenating examples from Flores (3-shot), GSM8K (2-shot), and TriviaQA (2-shot) [\(Joshi et al.,](#page-15-11) [2017\)](#page-15-11), in that order, with dotted lines indicating where a new task begins. Each column corresponds to a different context-switching prompt, created from distinct sampled inputs.

In both examples, BTS demonstrates its ability to dynamically adjust expert utilization. During the Flores prompt, the seed model and multilingual expert are predominantly active. During the GSM8K prompt, the math expert takes over, and finally, the seed model is most utilized for the TriviaQA prompt. This highlights BTS's capability to correctly activate the relevant experts for each task, even when transitioning between diverse contexts.

<span id="page-12-0"></span>![](_page_12_Figure_3.jpeg)

Figure 3 Visualization of the gate values of BTS's final stitch layer for context-switching sequences at inference time. These sequences are constructed by concatenating question-answer examples from Flores (3-shot), GSM8K (2-shot), and TriviaQA (2-shot), in that order, with dotted lines indicating task transitions. Each plot corresponds to a different randomly sampled prompt. This visualization highlights BTS's ability to dynamically adjust expert utilization based on token-level context.

# 5 Related work

Weights merging Previous works have demonstrated that linearly interpolating the weights of multiple expert models with the same architecture can produce a more effective model. Model Souping [\(Wortsman et al.,](#page-15-8) [2022\)](#page-15-8) achieves this by uniformly averaging model weights, whereas methods like BTM [\(Li et al.,](#page-15-0) [2022\)](#page-15-0), C-BTM [\(Gururangan et al.,](#page-14-2) [2023\)](#page-14-2), and SMEAR [\(Muqeeth et al.,](#page-15-12) [2023\)](#page-15-12) dynamically compute the weighting of each expert's model parameters based on the given prompt.

Output ensembles In addition to averaging model weights, several works have explored averaging model outputs to create ensembles of expert models [\(Li et al.,](#page-15-0) [2022;](#page-15-0) [Gururangan et al.,](#page-14-2) [2023\)](#page-14-2). Unlike BTS, these approaches do not require further training, and as such, they may be limited in expressivity.

Routing among dense models Another approach involves routing the entire input and generation to a single model selected from multiple expert LLMs [\(Filippova et al.,](#page-14-12) [2024;](#page-14-12) [Ong et al.,](#page-15-13) [2024\)](#page-15-13). However, these methods are limited when the input requires expertise from multiple domains or involves context-switching between different tasks. In contrast, BTS makes token-level decisions about combining experts, allowing it to seamlessly context switch across multiple tasks or adapt to inputs that require diverse or evolving skill sets.

Mixture-of-Experts upcycling Several works have explored using pretrained dense models to initialize Mixture of Experts (MoEs) [\(Komatsuzaki et al.,](#page-15-14) [2022;](#page-15-14) [Sukhbaatar et al.,](#page-15-3) [2024;](#page-15-3) [Zhang et al.,](#page-16-3) [2024\)](#page-16-3). These approaches copy each expert model's parameters to initialize the corresponding experts in the MoE. For the MoE's non-expert parameters, they average the parameters of the pretrained experts. The router is initialized from scratch. Following initialization, the MoE undergoes a training phase where all model parameters are updated. On the other hand, BTS only updates only the lightweight stitch layers, keeping all expert parameters frozen. Keeping experts intact after merging leads more more interpretable routing, and flexibility to add or remove experts with a small amount of additional training.

Adding connections between models Several recent works have proposed adapting language models to new modalities by composing modality-specific models, e.g., [Alayrac et al.](#page-14-13) [\(2022\)](#page-14-13) propose adding cross-attention parameters to allow a language model to condition on visual inputs, and [Liang et al.](#page-15-15) [\(2024\)](#page-15-15) uses global self-attention to fuse models for different modalities. Perhaps most similar to our work, [Bansal et al.](#page-14-14) [\(2024\)](#page-14-14) extend this idea to domain-specific language models, and propose augmenting an "anchor" language model with a single domain-specific model through cross-attention.

# 6 Conclusion

We introduced Branch-Train-Stitch, or BTS, a simple, flexible method for merging expert models to create a stronger, unified, generalist model. BTS combines expert models by inserting novel "stitch" layers between expert language model layers, which are learned in a lightweight training step. In experiments, we find that this approach outperforms competitive baselines, yielding the strongest generalist model performance with only a small number of training parameters. In some settings, BTS is shown to even outperform the expert models in their specialized domains. We further demonstrate that a BTS model can demonstrate new skills at the intersection of expert domains and motivate this architecture with extensive ablations and analysis. We hope this work furthers research into efficient and flexible methods for creating generalist large language models from modular, independently-trained experts.

# Acknowledgments

We extend our thanks to Sachin Mehta, Ruslan Mavlyutov, Alexei Baevski, Onur Çelebi, Niladri Chatterji, and Mik Vyatskov for their assistance with the training infrastructure. We thank Vedanuj Goswami for his assistance in training the models used in the experiments. We are also grateful to Anirudh Goyal, Akhil Mathur, and Wenhan Xiong for providing helpful pointers regarding data mixtures. We appreciate Ellen Tan, Rocky Wang, Todor Mihaylov, Xuchao Jia, and Mihir Sanjay Kale for helping with using the data preparation pipeline. We also thank Florian Laplantif, Norman Cheng, Lovish Madaan, and Kunal Bhalla for their support with the evaluation infrastructure. QZ thanks Chris Lu for discussions on the project. Finally, we thank Melanie Kambadur for her general assistance and support of this project.

# References

- <span id="page-14-1"></span>Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, and Sonal Gupta. Better fine-tuning by reducing representational collapse. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. [https://openreview.net/forum?id=](https://openreview.net/forum?id=OQ08SN70M1V) [OQ08SN70M1V](https://openreview.net/forum?id=OQ08SN70M1V).
- <span id="page-14-13"></span>Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. [http:](http://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html) [//papers.nips.cc/paper\\_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html).
- <span id="page-14-6"></span>Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021.
- <span id="page-14-14"></span>Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Shikhar Vashishth, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, and Partha Talukdar. Llm augmented llms: Expanding capabilities through composition. arXiv preprint arXiv:2401.02412, 2024.
- <span id="page-14-11"></span>Terra Blevins and Luke Zettlemoyer. Language contamination helps explain the cross-lingual capabilities of english pretrained models. arXiv preprint arXiv:2204.08110, 2022.
- <span id="page-14-7"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.
- <span id="page-14-9"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- <span id="page-14-0"></span>Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- <span id="page-14-12"></span>Anastasiia Filippova, Angelos Katharopoulos, David Grangier, and Ronan Collobert. No need to talk: Asynchronous mixture of language models. arXiv preprint arXiv:2410.03529, 2024.
- <span id="page-14-8"></span>Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc'Aurelio Ranzato, Francisco Guzmán, and Angela Fan. The flores-101 evaluation benchmark for low-resource and multilingual machine translation. Transactions of the Association for Computational Linguistics, 10:522–538, 2022.
- <span id="page-14-3"></span>Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A. Smith. Don't stop pretraining: Adapt language models to domains and tasks, 2020. <https://arxiv.org/abs/2004.10964>.
- <span id="page-14-2"></span>Suchin Gururangan, Margaret Li, Mike Lewis, Weijia Shi, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Scaling expert language models with unsupervised domain discovery. arXiv preprint arXiv:2303.14177, 2023.
- <span id="page-14-5"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021a.
- <span id="page-14-10"></span>Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021b.
- <span id="page-14-4"></span>Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

- <span id="page-15-11"></span>Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.
- <span id="page-15-14"></span>Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-of-experts from dense checkpoints. arXiv preprint arXiv:2212.05055, 2022.
- <span id="page-15-0"></span>Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. arXiv preprint arXiv:2208.03306, 2022.
- <span id="page-15-15"></span>Weixin Liang, Lili Yu, Liang Luo, Srinivasan Iyer, Ning Dong, Chunting Zhou, Gargi Ghosh, Mike Lewis, Wen-tau Yih, Luke Zettlemoyer, et al. Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models. arXiv preprint arXiv:2411.04996, 2024.
- <span id="page-15-1"></span>Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology of Learning and Motivation, pages 109–165. Academic Press, 1989. doi: https:// doi.org/10.1016/S0079-7421(08)60536-8. <https://www.sciencedirect.com/science/article/pii/S0079742108605368>.
- <span id="page-15-12"></span>Mohammed Muqeeth, Haokun Liu, and Colin Raffel. Soft merging of experts with adaptive routing. arXiv preprint arXiv:2306.03745, 2023.
- <span id="page-15-13"></span>Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E Gonzalez, M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data. arXiv preprint arXiv:2406.18665, 2024.
- <span id="page-15-7"></span>Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. Openwebmath: An open dataset of high-quality mathematical web text. arXiv preprint arXiv:2310.06786, 2023.
- <span id="page-15-6"></span>Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code. CoRR, abs/2308.12950, 2023. doi: 10.48550/ARXIV.2308.12950. <https://doi.org/10.48550/arXiv.2308.12950>.
- <span id="page-15-4"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. <https://openreview.net/forum?id=B1ckMDqlg>.
- <span id="page-15-10"></span>Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, et al. Language models are multilingual chain-of-thought reasoners. arXiv preprint arXiv:2210.03057, 2022.
- <span id="page-15-3"></span>Sainbayar Sukhbaatar, Olga Golovneva, Vasu Sharma, Hu Xu, Xi Victoria Lin, Baptiste Rozière, Jacob Kahn, Daniel Li, Wen-tau Yih, Jason Weston, et al. Branch-train-mix: Mixing expert llms into a mixture-of-experts llm. arXiv preprint arXiv:2403.07816, 2024.
- <span id="page-15-9"></span>Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.
- <span id="page-15-2"></span>Yi-Lin Tuan, Xilun Chen, Eric Michael Smith, Louis Martin, Soumya Batra, Asli Celikyilmaz, William Yang Wang, and Daniel M. Bikel. Towards safety and helpfulness balanced responses via controllable large language models. CoRR, abs/2404.01295, 2024. doi: 10.48550/ARXIV.2404.01295. <https://doi.org/10.48550/arXiv.2404.01295>.
- <span id="page-15-5"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. [https://proceedings.neurips.cc/paper/2017/hash/](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) [3f5ee243547dee91fbd053c1c4a845aa-Abstract.html](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
- <span id="page-15-8"></span>Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In International conference on machine learning, pages 23965–23998. PMLR, 2022.

- <span id="page-16-0"></span>Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. Doremi: Optimizing data mixtures speeds up language model pretraining. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. [http://papers.nips.cc/paper\\_files/paper/2023/hash/](http://papers.nips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html) [dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html).
- <span id="page-16-1"></span>Jiasheng Ye, Peiju Liu, Tianxiang Sun, Yunhua Zhou, Jun Zhan, and Xipeng Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. CoRR, abs/2403.16952, 2024. doi: 10.48550/ARXIV.2403. 16952. <https://doi.org/10.48550/arXiv.2403.16952>.
- <span id="page-16-3"></span>Qizhen Zhang, Nikolas Gritsch, Dwaraknath Gnaneshwar, Simon Guo, David Cairuz, Bharat Venkitesh, Jakob Foerster, Phil Blunsom, Sebastian Ruder, Ahmet Ustun, et al. Bam! just like that: Simple and efficient parameter upcycling for mixture of experts. arXiv preprint arXiv:2408.08274, 2024.
- <span id="page-16-2"></span>Xiaofeng Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong, and Zhang Xiong. Mixture of attention heads: Selecting attention heads per token. arXiv preprint arXiv:2210.05144, 2022.
- <span id="page-16-4"></span>Ming Zhong, Aston Zhang, Xuewei Wang, Rui Hou, Wenhan Xiong, Chenguang Zhu, Zhengxing Chen, Liang Tan, Chloe Bi, Mike Lewis, et al. Law of the weakest link: Cross capabilities of large language models. arXiv preprint arXiv:2409.19951, 2024.

# <span id="page-17-0"></span>Appendix

# A Cross capabilities: additional details and experiments

## A.1 Further experiment details

Russian expert model training To enhance cross capabilities in mathematical skills for Russian, we train an additional expert specifically on Russian data. The expert training setup follows the same procedure outlined in [Section 3.1.](#page-4-1) For training data, we utilize the Russian subset of the multilingual dataset previously used for the multilingual expert, as described in [Section 3.2.](#page-4-2)

Merged models training As an additional baseline, we continually pretrain the strongest dense model, the seed model, on the same Russian math pretraining data used for the merged models. All experiments share the following training configuration, except for BTM and Model Soup which does not require further training:

- Learning rate schedule: we warm up from 0 to 5e − 6 over 1000 steps, then undergoes a cosine decay to 10% of the peak learning rate. The merged models are trained for a total of 2000 steps. One exception is the expert routing model is trained for 1000 steps in total with a constant learning rate of 5e-4. This was chosen upon tuning the hyperparameters.
- Batch size: we use a batch size of 1M tokens.
- Token count: All models were trained on 2B tokens of Russian mathematics over 2000 training steps. The exception is expert routing, which only trained on 1B tokens over 1000 steps, as we did not see performance improvement with further training.

## <span id="page-17-1"></span>A.2 Results on merging and upcycling models without in-domain data

|                  |       |       | Flores |           |
|------------------|-------|-------|--------|-----------|
|                  | GSM8K | En/Ru | Ru/En  | Ru-MGSM   |
| Dense models     |       |       |        |           |
| Seed Model       | 10.5  | 22.8  | 32.8   | 12.8      |
| Math Expert      | ∗20.5 | 10.2  | 28.9   | 10.8      |
| Russian Expert   | 9.48  | ∗32.3 | ∗34.6  | 9.60      |
| Expert upcycling |       |       |        |           |
| BTX Sample       | 18.3  | 30.4  | 34.0   | 10.0      |
| BTX Soft         | 18.0  | 30.0  | 33.9   | 12.4      |
| BAM              | ∗20.5 | 30.6  | 34.5   | 10.8      |
| Expert merging   |       |       |        |           |
| Model Soup       | 17.5  | 14.7  | 32.3   | ∗<br>13.2 |
| BTM              | ∗20.5 | ∗32.3 | ∗34.6  | 9.60      |
| Expert Routing   | ∗20.5 | 32.3  | ∗34.6  | 9.60      |
| BAM Adapter      | 18.1  | 30.9  | 34.1   | 12.8      |
| BTS              | 19.0  | 31.6  | 33.0   | 10.0      |

Table 9 Cross capability performance of merged models without in-domain data. We evaluate the seed model, Russianlanguage, and Math experts on Russian MGSM [\(Shi et al.,](#page-15-10) [2022\)](#page-15-10) and compare performance with merged and upcycled models. We do not use any in-domain training data during the merging or upcycling training process. The results indicate that a small amount of cross capability data is necessary for merged or upcycled models to effectively learn cross capabilities.

In [Table 9,](#page-17-1) we show results on merging and upcycling models without in-domain data. The merging phase is instead trained on a data mixture composed of 50% of math expert and 50% of Russian expert's continue pretraining data mixture.

We observe that despite being trained with more tokens during the merging phase, all baseline methods does not significantly outperform the seed model on the cross capability task Russian MGSM. This indicates that in-distribution data is essential.

# <span id="page-18-0"></span>B Pseudo Code for BTS

```
d e f S ti t c h L a y e r ( xs , merge_into_hub=True ) :
    """
     xs : dense models ' outputs
    """
     x_hub = x [ 0 ]
     x_experts = x [ 1 : ]
     g = w_gate ( x_hub ) # [ bs , seq_len , dim, 1+n_experts ]
    # Experts - into -Hub Layer
     i f merge_into_hub :
          g = dropout ( g ) . so ftmax ( dim=- 1 )
          h_experts = [
               w_proj [ i ] ( x_experts [ i ] ) f o r i i n r an ge ( n_experts )
          h_hub = ( g * s t a c k ( [ h ] + h_experts , dim=- 1 ) ) . sum( - 1 )
    # Merge- into -Expert Layer
     e l s e :
          g = dropout ( g ) . si gm oid ( )
          h_experts = = [
                     ( 1 - g [ . . . , i + 1 ] ) * x_experts [ i ]
                    + ( g [ . . . , i + 1 ] * w_proj [ i ] ( x_hub ) )
                     f o r i i n r an ge ( n_experts )
          h_hub = x_hub
     r e t u r n s t a c k ( [ h_hub ] + h_experts , dim=- 1 )
d e f BTSBlock ( xs , i t h_l a y e r , BTS_freq ) :
     x_hub = hub_model_layer ( xs [ 0 ] )
     x_experts = [ expert_model_layer [ i ] ( xs [ i+1 ] ) f o r i i n n_experts ]
     xs = s t a c k ( [ x_hub ] + x_experts , dim=- 1 )
     i f i t h_l a y e r % BTS_freq = = 0 :
         # Alternate between two types of s t i t ch layers
          hs = S ti t c h L a y e r ( xs , merge_into_hub=( i t h_l a y e r // BTS_freq )%2 )
          r e t u r n hs
     r e t u r n xs
```