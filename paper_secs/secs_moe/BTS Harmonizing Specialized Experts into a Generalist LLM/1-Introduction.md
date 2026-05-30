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

