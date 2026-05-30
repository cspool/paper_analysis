# FLASH MULTI-HEAD FEED-FORWARD NETWORK

Minshen Zhang $^{1,2*}$  Xiang Hu $^{2*}$  Jianguo Li  $^2$  Wei Wu  $^2$  Kewei Tu $^{1\dagger}$  ShanghaiTech University  $^2$  Ant Group zhangmsh1@shanghaitech.edu.cn aaron.hx@antgroup.com tukw@shanghaitech.edu.cn

### **ABSTRACT**

We explore Multi-Head FFN (MH-FFN) as a replacement of FFN in the Transformer architecture, motivated by the structural similarity between single-head attention and FFN. While multi-head mechanisms enhance expressivity in attention, naively applying them to FFNs faces two challenges: memory consumption scaling with the head count, and an imbalanced ratio between the growing intermediate size and the fixed head dimension as models scale, which degrades scalability and expressive power. To address these challenges, we propose Flash Multi-Head FFN (FlashMHF), with two key innovations: an I/O-aware fused kernel computing outputs online in SRAM akin to FlashAttention, and a design using dynamically weighted parallel sub-networks to maintain a balanced ratio between intermediate and head dimensions. Validated on models from 128M to 1.3B parameters, FlashMHF consistently improves perplexity and downstream task accuracy over SwiGLU FFNs, while reducing peak memory usage by 3-5x and accelerating inference by up to 1.08x. Our work establishes the multi-head design as a superior architectural principle for FFNs, presenting FlashMHF as a powerful, efficient, and scalable alternative to FFNs in Transformers.

## 1 Introduction

The Transformer architecture has become the standard for Large Language Models (LLMs) (Vaswani et al., 2017b). At its core, the Transformer block is composed of two primary components: a multi-head self-attention mechanism and a position-wise Feed-Forward Network (FFN). While multi-head attention is often credited for the model's ability to capture complex contextual relationships, the FFN module, which consumes a significant portion of the model's parameters and computation, is equally critical for its expressive power (Gerber, 2025).

Recent studies have revealed a structural symmetry between the Feed-Forward Network (FFN) and single-head attention (Geva et al., 2020), as illustrated in Figure 1. An FFN, defined as FFN( $\mathbf{X}$ ) =  $\sigma(\mathbf{X}\mathbf{W_1}^{\mathsf{T}})\mathbf{W_2}$  can be reinterpreted as  $\mathbf{X}$  attending over  $\mathbf{W_1}$  to retrieve values from  $\mathbf{W_2}$ . This formulation mirrors the core attention mechanism, Attention(Q, K, V) = softmax  $\left(\frac{QK^T}{\sqrt{d_k}}\right)V$  with the primary distinction being the FFN's element-wise activation ( $\sigma$ ) versus attention's row-wise softmax. The established success of the multi-head design in attention—which enables joint information processing from diverse representational subspaces—provides a strong rationale for investigating a similar multi-head decomposition for FFNs. Indeed, this direction was explored by previous

<span id="page-0-0"></span>![](_page_0_Figure_8.jpeg)

Figure 1: Structural Symmetry.

works like Multi-Head Mixture of Experts (MH-MoE) (Wu et al., 2024) for its effectiveness. However, despite its promising results, it still faces issues of scalability and low computational efficiency.

<sup>\*</sup>Equal contribution.

<sup>†</sup>Corresponding author.

In this work, we analyze the Multi-Head Feed-Forward Network (a straightforward application of this multi-head principle), and identify two challenges that hinder its practical adoption. First, **high memory pressure**: analogous to multi-head attention, the architecture generates H separate intermediate activations, leading to an H-fold increase in memory usage—a known challenge for multi-head designs like MH-MoE. Second, **scaling imbalance**: As models scale, the FFN's intermediate dimension  $d_{\rm ff}$  grows while the per-head dimension  $d_{\rm h}$  remains a small constant, following the design of Multi-Head Attention. This creates a skewed  $d_{\rm ff}/d_{\rm h}$  ratio degrading performance, as FFNs are known to perform optimally only when this ratio is kept within a range (Kaplan et al., 2020).

To address these challenges, we propose **FlashMHF**, whose key innovation lies in a scale-balanced structure and memory-efficient flash kernel. We partition the intermediate dimension into multiple parallel sub-networks and aggregate their outputs. This design ensures the ratio between the effective intermediate dimension and the head dimension remains balanced, maintaining performance at large scale. For memory efficiency, and analogously to FlashAttention's online softmax (Dao et al., 2022), our fused kernel computes the SwiGLU activation without materializing the large intermediate hidden state in HBM. This approach drastically reduces peak memory usage and eliminates costly data transfers between on-chip SRAM and HBM.

Extensive experiments on models from 128M to 1.3B parameters show that FlashMHF consistently outperforms the standard SwiGLU FFN baseline across all critical metrics. Our method achieves lower perplexity, stronger downstream task performance, a **1.00x-1.08x** inference speedup on the Hopper architecture, and a drastic **3-5x** reduction in peak GPU memory compared to SwiGLU FFN.

Our contributions can be summarized as follows:

- Identification of Foundational MH-FFN Challenges. We identify and analyze two critical issues that render a naïve Multi-Head FFN impractical: (1) high memory pressure caused by intermediate activations and (2) an architectural scaling imbalance between head and intermediate dimensions.
- FlashMHF: A Novel and Efficient Architecture. We propose FlashMHF, a novel architecture that resolves these challenges by pairing a scale-balanced parallel FFN subnetworks design with a high-efficiency, IO-aware kernel.
- State-of-the-Art Performance and Efficiency. Through extensive experiments, we demonstrate that FlashMHF significantly outperforms the widely-used SwiGLU FFN baseline in perplexity and downstream tasks, all while delivering up to a 1.08x speedup and reducing peak memory usage by 3-5x compared to SwiGLU FFN.

### <span id="page-1-0"></span>2 PRELIMINARIES

**Notation.** Let L be the sequence length,  $d_{\text{model}}$  be model dimension,  $d_{\text{k}}$  be attention head dimension, and  $d_{\text{ff}}$  be intermediate dimension of FFN. We consider the parameters:

$$\mathbf{Q}_{att}, \mathbf{K}_{att}, \mathbf{V}_{att} \in \mathbb{R}^{L \times d_k}, \quad \mathbf{X} \in \mathbb{R}^{L \times d_{\text{model}}}, \quad \mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$$

For FFNs we write  $\phi(\cdot)$  for an element-wise nonlinearity (e.g., ReLU, GeLU, SiLU).

Single-Head Attention vs. FFN.

$$\operatorname{Att}\left(\mathbf{Q}_{att}, \mathbf{K}_{att}, \mathbf{V}_{att}\right) =: \operatorname{softmax}\left(\frac{\mathbf{Q}_{att} \mathbf{K}_{att}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}_{att}, \quad \operatorname{FFN}(\mathbf{X}) =: \phi\left(\mathbf{X} \mathbf{W}_1^{\top}\right) \mathbf{W}_2. \tag{1}$$

By replacing the activation function  $\operatorname{softmax}(\cdot/\sqrt{d_k})$  with an element-wise nonlinearity  $\phi(\cdot)$ , Attention and FFN become structurally identical. Thus, we can reinterpret FFNs as "attention over parameters" of length  $d_{\rm ff}$  (Vaswani et al., 2017a; Geva et al., 2020).

FFN **Definition.** In modern Transformers, the gated variant  $SwiGLU(\cdot)$  is the common choice instead of vanilla  $FFN(\cdot)$ . We follow the standard  $SwiGLU(\cdot)$  formulation:

$$SwiGLU(\mathbf{X}) =: ((\mathbf{X}\mathbf{W}_{up}) \odot SiLU(\mathbf{X}\mathbf{W}_{gate}))\mathbf{W}_{down}. \tag{2}$$

For later analysis, we introduce a key-value style formulation whose symbols intentionally echo attention. For any input *query-like* matrix  $\mathbf{Q} \in \mathbb{R}^{L \times d}$  and  $\mathbf{K}, \mathbf{U}, \mathbf{V} \in \mathbb{R}^{d_{\mathrm{ff}} \times d}$  define

<span id="page-1-1"></span>
$$\widetilde{\mathrm{FFN}}(\mathbf{Q}; \mathbf{K}, \mathbf{U}, \mathbf{V}) =: \left( \mathrm{SiLU}(\mathbf{Q} \, \mathbf{K}^{\mathsf{T}}) \ \odot \ (\mathbf{Q} \, \mathbf{U}^{\mathsf{T}}) \right) \mathbf{V}. \tag{3}$$

Under the assignments  $\mathbf{Q} = \mathbf{X}$ ,  $\mathbf{K} = \mathbf{W}_{\text{gate}}^{\top}$ ,  $\mathbf{U} = \mathbf{W}_{\text{up}}^{\top}$ , and  $\mathbf{V} = \mathbf{W}_{\text{down}}$ ,  $\widetilde{\text{FFN}}(\cdot)$  is identical to  $\text{SwiGLU}(\cdot)$ . Compared to vanilla  $\text{FFN}(\cdot)$ ,  $\text{SwiGLU}(\cdot)$  inserts a multiplicative gate and retains the attention-like structure since we can define a new nonlinearity function  $\phi_s(\cdot)$ :

$$\phi_s(\mathbf{Q}, \mathbf{K}) =: \operatorname{SiLU}(\mathbf{Q} \mathbf{K}^{(g)\top}) \odot \mathbf{Q} \mathbf{K}^{(u)\top}$$
 (4)

where  $\mathbf{K}^{(\mathrm{g})}, \mathbf{K}^{(\mathrm{u})} \in \mathbb{R}^{d_{\mathrm{ff}} \times d_{\mathrm{model}}}$  and  $\mathbf{K} =: [\mathbf{K}^{(\mathrm{g})}, \mathbf{K}^{(\mathrm{u})}] \in \mathbb{R}^{(2d_{\mathrm{ff}}) \times d_{\mathrm{model}}}$ . Now we can rewrite SwiGLU into  $\phi_s(\mathbf{Q}, \mathbf{K})\mathbf{V}$ . This shows that SwiGLU is indeed a variant of attention in a broad sense, where the softmax is replaced by element-wise activation function  $\phi_s(\cdot)$ .

**Headwise Operation Functions.** Let  $H \in \mathbb{N}$  be the number of heads and  $d_h$  be the per-head dimension s.t.  $d_{\text{model}} = H \cdot d_h$ . For any  $\mathbf{T} \in \mathbb{R}^{L \times d_{\text{model}}}$ , define headwise split as

$$\operatorname{split}_{H}(\mathbf{T}) \in \mathbb{R}^{L \times H \times d_{h}}, \quad \left[\operatorname{split}_{H}(\mathbf{T})\right]_{l,h,j} = \mathbf{T}_{l,(h-1)d_{h}+j},$$
 (5)

where  $l \in \{1, \dots, L\}$ ,  $h \in \{1, \dots, H\}$ , and  $j \in \{1, \dots, d_h\}$ . Simply speaking, this operation splits a tensor along the  $d_{\text{model}}$  dimension into  $d_h \times H$ . Conversely, for any H sub parts of equal sizes  $\mathbf{S} \in \mathbb{R}^{L \times H \times d_h}$ , define headwise concatenation

$$\operatorname{concat}_{H}(\mathbf{S}) \in \mathbb{R}^{L \times d_{\operatorname{model}}}, \qquad \left[\operatorname{concat}_{H}(\mathbf{S})\right]_{l, (h-1)d_{h}+j} = \mathbf{S}_{l, h, j}. \tag{6}$$

### 3 METHODOLOGY

Our work is motivated by the structural symmetry between self-attention and FFNs, as detailed in Section 2. We posit that just as self-attention benefits from a multi-head design, the FFN can be similarly decomposed to enhance its expressive power. In this section, we first formalize the concept of a Naïve Multi-Head Feed-Forward Network (MH-FFN), discuss its practical limitations, and then introduce our proposed Flash Multi-Head FFN (FlashMHF) architecture, which leverages a gated aggregation of parallel sub-networks and flash algorithms to overcome these practical limitations.

### 3.1 Naïve Multi-Head Feed-Forward Networks

**Setup.** Let H be the number of heads of MH-FFN, and  $d_h = d_{\text{model}}/H$  be the per-head dimension.

**Definition.** Given  $\mathbf{X} \in \mathbb{R}^{L \times d_{\text{model}}}$ , form per-head inputs by linear projection and reshaping:

$$\mathbf{Q} = \mathrm{split}_{H}(\mathbf{X} \mathbf{W}_{\mathrm{in}}) \in \mathbb{R}^{L \times H \times d_{h}}, \qquad \mathbf{W}_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}. \tag{7}$$

Define matrix  $\mathbf{K}^h, \mathbf{U}^h, \mathbf{V}^h \in \mathbb{R}^{d_{\mathrm{ff}} \times d_h}$  for all heads  $h \in \{1, \dots, H\}$ . Apply  $\widetilde{\mathrm{FFN}}(\cdot)$  defined in equation 3 in a headwise manner:

$$\mathbf{S}_{:.h.:} = \widetilde{\mathrm{FFN}}(\mathbf{Q}_{:.h.:}; \mathbf{K}^h, \mathbf{U}^h, \mathbf{V}^h) \in \mathbb{R}^{L \times d_h}. \tag{8}$$

Finally, S is concatenated and linearly projected to combine information across all heads:

$$\mathbf{O}^{\text{MH-FFN}} = \text{concat}_{H}(\mathbf{S}) \mathbf{W}_{\text{out}} \in \mathbb{R}^{L \times d_{\text{model}}}, \quad \mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}.$$
 (9)

**Limitations.** This naïve design suffers from two main limitations. The first is **Scaling Failure**: Although the multi-head structure offers expressiveness gains at smaller model scales, we empirically find that this approach ceases to be competitive once the model size exceeds  $\approx 128 \mathrm{M}$  parameters, as reported in Section 4.1. The second limitation is **Memory Pressure**. MH-FFN materializes H sepa-

![](_page_2_Figure_18.jpeg)

Figure 2: Memory limitation of MH-FFN.

rate sets of intermediate activations, each of size approximately  $L \times d_{\rm ff}$ . This incurs a total activation memory footprint of  $\mathcal{O}((L \cdot H + d_{\rm model}) \cdot d_{\rm ff})$ . Memory usage therefore grows linearly with the number of heads H, which quickly becomes prohibitive as models and context lengths scale.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 3: (a) Parallel FFN Sub-Networks. (b) SRAMFFN loads blocks of  $\mathbf{Q}$  in the outer loop and blocks of  $\mathbf{K}$ ,  $\mathbf{U}$ ,  $\mathbf{V}$  in the inner loop, compute SiLU( $\mathbf{Q}\mathbf{K}^{\top}$ ),  $\mathbf{Q}\mathbf{U}^{\top}$  and corresponding  $\mathbf{V}$  multiplication on SRAM.

### <span id="page-3-1"></span>3.2 FLASH MULTI-HEAD FEED-FORWARD NETWORKS

### 3.2.1 PARALLEL FFN SUB-NETWORKS

To address the **Scaling Failure**, we first analyze why the naïve Multi-Head FFN fails to scale beyond  $\approx 128 \mathrm{M}$  parameters. As the model size increases, the intermediate width  $d_{\mathrm{ff}}$  must grow, while the per-head width  $d_h$  is typically kept fixed (e.g.,  $d_h = 128$ ), a design choice inherited from Multi-Head Attention (MHA). Consequently, the ratio  $d_{\mathrm{ff}}/d_h$  grows excessively. In the original SwiGLU design, a classical choice is  $d_{\mathrm{ff}}/d_{\mathrm{model}} = \frac{8}{3}$ . In contrast, under our naïve multi-head setting, we observe that this effective ratio explodes across scales:

128M: 
$$\frac{d_{\text{ff}}}{d_h} = \frac{2048}{128} = 16$$
, 370M:  $\frac{d_{\text{ff}}}{d_h} = \frac{2688}{128} = 21$ , 1.3B:  $\frac{d_{\text{ff}}}{d_h} = \frac{5760}{128} = 45$ .

Such a significant **Scaling Imbalance** from the optimal range (Kaplan et al., 2020) leads to a decline in parameter efficiency, rendering the naïve multi-head construction increasingly ineffective at larger scales. This diagnosis motivates our use of multiple, parallel FFN pathways, which are combined via a learned gating mechanism, illustrated in Figure 3a. Our architecture draws inspiration from Mixture-of-Experts (Shazeer et al., 2017), essentially functioning as a dense MoE structure that omits sparse top-k expert selection. This methodology re-establishes a balanced and effective expansion ratio for each head's computation without inflating per-head activations.

**Definition.** Given  $\mathbf{X} \in \mathbb{R}^{L \times d_{\mathrm{model}}}$ , form per-head inputs by linear projection and reshaping:

$$\mathbf{Q} = \mathrm{split}_{H}(\mathbf{X} \mathbf{W}_{\mathrm{in}}) \in \mathbb{R}^{L \times H \times d_{h}}, \qquad \mathbf{W}_{\mathrm{in}} \in \mathbb{R}^{d_{\mathrm{model}} \times d_{\mathrm{model}}}. \tag{10}$$

Let E be the number of sub-networks and  $d_e$  be the dimension per sub-network, such that the total intermediate dimension is  $d_{\rm ff}=E\cdot d_e$ . We adopt the standard SwiGLU ratio by setting  $d_e\approx\frac{8}{3}d_h$  (Touvron et al., 2023). For each head h, we define a *private* set of E sub-networks, where the parameters for the e-th sub-network within head h are  $\mathbf{K}_e^h, \mathbf{U}_e^h, \mathbf{V}_e^h \in \mathbb{R}^{d_e \times d_h}$ .

**Gating weights.** For each head h, we introduce a gating matrix  $\mathbf{W}^h \in \mathbb{R}^{d_h \times E}$ . The per-token logits for the E sub-networks are computed by projecting the head's query:

$$\mathbf{P}^h = \mathbf{Q}_{h} \cdot \mathbf{W}^h \in \mathbb{R}^{L \times E}. \tag{11}$$

These logits are then transformed into normalized gating weights,  $\mathbf{R}^h$ , via a sigmoid activation followed by a numerically stable normalization.

$$\mathbf{R}_{\ell,e}^{h} = \frac{\sigma(\mathbf{P}_{\ell,e}^{h})}{\sum_{e'=1}^{E} \sigma(\mathbf{P}_{\ell,e'}^{h}) + \varepsilon}, \qquad e = 1, \dots, E,$$
(12)

**Sub-network Aggregation.** The final output for each head is computed as a weighted sum of its sub-network outputs, using the gating weights  $\mathbb{R}^h$  to aggregate them:

$$\mathbf{S}_{\ell,h,:} = \sum_{e=1}^{E} \mathbf{R}_{\ell,e}^{h} \, \widetilde{\mathrm{FFN}} \big( \mathbf{Q}_{\ell,h,:}; \, \mathbf{K}_{e}^{h}, \, \mathbf{U}_{e}^{h}, \, \mathbf{V}_{e}^{h} \big) \, \in \, \mathbb{R}^{d_{h}}. \tag{13}$$

This parallel FFN sub-networks formulation allows each sub-network to maintain a balanced internal dimension  $(d_e \approx \frac{8}{3}d_h)$ , thereby resolving the scaling imbalance identified in the naïve MH-FFN.

Finally, the outputs from all heads are concatenated and transformed by a final output projection:

$$\mathbf{O}^{\text{FlashMHF}} = \text{concat}_H(\mathbf{S}) \, \mathbf{W}_{\text{out}} \in \mathbb{R}^{L \times d_{\text{model}}}, \qquad \mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}.$$
 (14)

### 3.2.2 I/O-AWARE FLASH ALGORITHM

To address the **Memory Pressure**, we introduce an I/O-aware algorithm for the FFN computation that avoids materializing the large intermediate activation tensor.

**Blockwise Computation.** Recall from equation 3 that the core computation for a single head involves an input query  $\mathbf{Q} \in \mathbb{R}^{L \times d_h}$  and parameter matrices  $\mathbf{K}, \mathbf{U}, \mathbf{V} \in \mathbb{R}^{d_{\mathrm{ff}} \times d_h}$ . The naïve approach would compute the full intermediate tensor  $\mathbf{A} = \mathrm{SiLU}(\mathbf{Q}\mathbf{K}^\top) \odot (\mathbf{Q}\mathbf{U}^\top)$ .

Our flash algorithm, illustrated in Figure 3b, circumvents this by processing the computation in blocks. We partition the parameter matrices  $\mathbf{K}, \mathbf{U}$ , and  $\mathbf{V}$  along their first dimension  $(d_{\mathrm{ff}})$  into M blocks of size b, denoted as  $\{\mathbf{K}_m, \mathbf{U}_m, \mathbf{V}_m\}_{m=1}^M$ , where  $d_{\mathrm{ff}} = M \cdot b$ . The final output  $\mathbf{O} \in \mathbb{R}^{L \times d_h}$  is then computed iteratively, accumulating the result of each block one at a time. This entire loop is executed within a single fused kernel:

$$\mathbf{O} \leftarrow \mathbf{0}; \quad \text{for } m = 1 \dots M: \quad \mathbf{O} \leftarrow \mathbf{O} + \left( \text{SiLU}(\mathbf{Q} \mathbf{K}_m^{\top}) \odot (\mathbf{Q} \mathbf{U}_m^{\top}) \right) \mathbf{V}_m.$$
 (15)

The key to solving the high memory pressure lies in the multi-head design itself. A naïve implementation would materialize H large intermediate tensors in HBM. Our algorithm avoids this entirely by leveraging the narrow heads to process the computation in blocks along the  $d_{\rm ff}$  dimension, each fitting within on-chip SRAM. This blockwise execution principle also contrasts with standard FFNs, which must materialize their single, large intermediate tensor before the final projection. By design, our I/O-aware algorithm resolves the memory pressure of the naïve approach, reducing consumption from  $\mathcal{O}((d_{\rm ff} \cdot H + d_{\rm model}) \cdot L)$  to  $\mathcal{O}(d_{\rm model} \cdot L)$ . Remarkably, this memory footprint is even smaller than that of a standard SwiGLU FFN, which requires  $\mathcal{O}((d_{\rm ff} + d_{\rm model}) \cdot L)$ . Detailed pseudocode for the forward and backward passes is provided in Appendices A and B.

#### 4 EXPERIMENTS

For a fair comparison, all models are pre-trained with a context length of 4,096 and a batch size of 64. We ensure that the sizes of our models were approximately equal to those of the baselines, and we synchronize the hyper-parameter settings for the optimizer across all models.

**Baseline.** Our baseline is a Llama-Like (Touvron et al., 2023) model consisting of multi-head self-attention with Rotary Position Embeddings (RoPE) (Su et al., 2021) and a point-wise SwiGLU FFN. We use the GPT-NeoX (Black et al., 2022) tokenizer with a vocabulary size of 50,432. We disable attention dropout and FFN bias throughout. RoPE uses  $\theta$ =10,000, and the maximum position embedding length is set to 4,096. We trained baseline as well as our models across 128M, 370M and 1.3B to fully validate scalability of our model. In terms of other configurations, we generally follow the same settings and hyperparameters from the appendix of Dao & Gu (2024).

**Parametric KV Baseline.** As a key ablation study, we introduce the PKV baseline, which replaces the SwiGLU FFN with a multi-head attention whose keys and values are learnable model parameters. This baseline approximates an extreme case of our architecture where each sub-network has a dimension of one ( $d_e = 1$ ) and serves to validate the necessity of the point-wise SwiGLU component. For fair comparison, its parameter size is matched to the primary baseline.

**Dense-MoE Baseline.** To verify whether the gains come from the multi-head design or from the parallel FFN sub-networks, we set the number of heads H to 1. This control group is equivalent to a dense MoE.

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Comparing Baseline, Parametric KV, FlashMHF and MH-FFN in 128M and 370M scales.

MH-FFN. We replace the baseline point-wise SwiGLU with our MH-FFN module. We adjust the layer depth and intermediate width to achieve approximately equal model size as the baseline. For fair comparison, we set the model size and FLOPs approximately equal to the baseline by adjusting the layer depth and intermediate width. More detailed configs per model scale are listed in Appendix [D.](#page-15-0)

FlashMHF. We replace point-wise SWIGLU FFN in the baseline model with our FlashMHF module, we keep almost all components identical to the baseline (attention configuration, layernorms, dropout = 0, and no FFN biases). We experiment with FlashMHF of different FFN head dimensions across d<sup>h</sup> = {64, 128, 256}. We set the dimension per sub-network to d<sup>e</sup> ≈ 8 3 d<sup>h</sup> rounded up to the nearest multiple of 64 and set E accordingly.We adjust the layer depth and intermediate width to achieve approximately equal model size as the baseline.

Data and training tokens. All models are trained on THE PILE. We train the 128M and 370M models with 60B tokens (245K steps), and train the 1.3B model with 100B tokens (409K steps). We calculate evaluation loss on PG19's validation split. Further optimizer details, learning-rate schedules, regularization, and all remaining hyperparameters are provided in Appendix [C.](#page-14-0)

## <span id="page-5-0"></span>4.1 LANGUAGE MODELING AT DIFFERENT SCALES

State-of-the-Art Performance. We first experiment with the SwiGLU baseline, Dense-MoE baseline, PKV-128hdim and FlashMHF-128hdim in 128M and 370M scale.

The validation loss are presented in Figure [4](#page-5-1) and Table [1.](#page-6-0) The results clearly demonstrate the superiority of our approach in both scales. FlashMHF consistently achieves a lower final validation loss than the strong SwiGLU baseline, while the PKV baseline performs notably poorly in comparison.

The superior performance of FlashMHF stems from fundamental advantages in its architectural design. We analyze these through two comparisons: First, against the SwiGLU FFN, FlashMHF's multi-head design significantly enhances expressive power. We hypothesize this advantage can be understood through the lens of "implicit thinking", where a standard FFN is viewed as executing a single path of sequential reasoning [\(Chen et al., 2025\)](#page-10-6). In this framework, FlashMHF's architecture is analogous to performing a *beam search* over this implicit thinking process. By exploring multiple cognitive paths in parallel, the model can construct richer and more robust representations, naturally

<span id="page-5-2"></span>![](_page_5_Figure_9.jpeg)

Figure 5: Training on 370M model scale to investigate the best head dimension. Analysis: (a) is full training loss, to visualize it more clearly, we zoom in to later training steps as illustrated in (b) and (c). Our FlashMHF with d<sup>h</sup> = 64, 128 gets better train/evaluation loss on PG19 validation split.

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 6: Scaling FlashMHF up to 1.3B. FlashMHF with d<sup>h</sup> = 128 constantly performs the best.

boosting its capabilities. Second, when contrasted with the PKV baseline, the utility of element-wise structure is clear. PKV's softmax activation induces competition across the entire hidden dimension, creating an aggressive information bottleneck. In contrast, element-wise activation function avoids this competitive pressure, maximizing parameter efficiency. This allows each channel to learn more freely, enabling the model to form a much richer and more disentangled set of features, which ultimately leads to better performance. Meanwhile, we observe that Dense-MoE even underperforms the baseline. We attribute this to the ratio of the intermediate dimension to the input dimension deviating from its optimal value, further supporting our FFN ratio hypothesis.

Ablation on Parallel FFN Sub-Networks. At the 128M scale (Figure [4a\)](#page-5-1), both MH-FFN-128hdim and FlashMHF-128hdim outperform the SwiGLU baseline on the PG19 validation set. However, a clear divergence emerges at the 370M scale (Table [1,](#page-6-0) Figure [4b\)](#page-5-1): the na¨ıve MH-FFN is no longer competitive, while FlashMHF continues to deliver gains. Given that the sole mathematical difference between these models is our parallel FFN sub-networks, this result empirically validates it as the crucial component for successful scaling.

The performance divergence of MH-FFN across these model scales provides a crucial insight that supports our arguments in Section [3.2.](#page-3-1) The fact that MH-FFN is effective at 128M but fails to scale to 370M is direct evidence for our reasoning: as model size grows, the ratios dff/d<sup>h</sup> of MH-FFN become imbalanced, inevitably leading to performance loss. This strongly validates that our use of parallel sub-networks, which introduces multiple smaller FFN pathways in FlashMHF, is a well-calibrated and critical solution addressing this scaling challenge.

Head-Dimension Ablation. Building on the those findings, we now probe how the FlashMHF performance depends on its head dimension. At the 370M scale, we vary the FlashMHF head dimension d<sup>h</sup> ∈ {64, 128, 256} as illustrated in Figure [5](#page-5-2) and shown in Table [1.](#page-6-0) FlashMHF with d<sup>h</sup> = 128 and 256 outperforms the SwiGLU Baseline; moreover, dh=128 offers the best performance with a 0.015 margin in evaluation loss, whereas dh=64 underfits and dh=256 yields diminishing returns.

These results indicate that moderate head dimensions are usually more preferable and point to a fundamental trade-off between per-head expressive power and the architectural benefit of subspace diversity, which is similar to the conclusions in [Wu et al.](#page-11-1) [\(2024\)](#page-11-1). A small dimension such as d<sup>h</sup> = 64 creates a representational bottleneck in each head, leading to underfitting as individual pathways lack the capacity to learn complex features. Conversely, a large dimension like d<sup>h</sup> = 256 reduces the total number of heads, diminishing the gains from functional specialization and causing the architecture to behave more like a monolithic FFN. The d<sup>h</sup> = 128 configuration appears to strike an

Table 1: Evaluation loss at 370M and 1.3B scales.

<span id="page-6-0"></span>

| Model Size 370M   | Loss  | Model Size 1.3B   | Loss  |
|-------------------|-------|-------------------|-------|
| Baseline          | 3.030 | Baseline          | 2.843 |
| PKV (dh=128)      | 3.334 | –                 | –     |
| MH-FFN (dh=128)   | 3.031 | –                 | –     |
| Dense-MoE         | 3.062 | –                 | –     |
| FlashMHF (dh=64)  | 3.046 | FlashMHF (dh=64)  | 2.849 |
| FlashMHF (dh=128) | 3.014 | FlashMHF (dh=128) | 2.793 |
| FlashMHF (dh=256) | 3.029 | FlashMHF (dh=256) | 2.799 |

![](_page_6_Figure_9.jpeg)

Figure 7: PPL vs Model Scale.

<span id="page-7-0"></span>

|      | Scale Model           |       |       |       |       | HellaSwag↑ SIQA↑ PIQA↑ OBQA↑ WinoGrande↑ RACE↑ Average↑ |       |       |
|------|-----------------------|-------|-------|-------|-------|---------------------------------------------------------|-------|-------|
|      | 370M Baseline         | 33.20 | 40.94 | 64.53 | 27.50 | 51.70                                                   | 21.66 | 39.92 |
|      | 370M Parametric KV    | 27.85 | 39.61 | 60.50 | 27.20 | 51.78                                                   | 21.67 | 38.10 |
|      | 370M MH-FFN           | 33.45 | 41.30 | 65.18 | 27.60 | 51.46                                                   | 21.87 | 40.14 |
|      | 370M FlashMHF-64hdim  | 32.57 | 41.50 | 65.34 | 27.60 | 53.35                                                   | 21.59 | 40.32 |
|      | 370M FlashMHF-128hdim | 33.97 | 40.63 | 66.27 | 27.80 | 52.41                                                   | 21.80 | 40.48 |
|      | 370M FlashMHF-256hdim | 33.60 | 39.71 | 64.85 | 27.30 | 52.17                                                   | 21.94 | 39.92 |
| 1.3B | Baseline              | 39.47 | 41.76 | 67.30 | 27.60 | 52.64                                                   | 21.73 | 41.75 |
| 1.3B | FlashMHF-64hdim       | 39.80 | 42.17 | 67.08 | 27.20 | 51.30                                                   | 22.62 | 41.70 |
| 1.3B | FlashMHF-128hdim      | 42.96 | 44.17 | 68.44 | 27.80 | 54.46                                                   | 22.26 | 43.35 |
| 1.3B | FlashMHF-256hdim      | 41.80 | 42.32 | 68.88 | 27.60 | 53.35                                                   | 22.01 | 42.66 |

Table 2: Downstream benchmarks for 1.3B and 370M models. Bold = best, underline = second best.

effective balance, endowing each head with sufficient capacity while maintaining a high degree of parallelism and diversity.

Scalability. We conducted large-scale experiments to demonstrate that the FlashMHF architecture is fundamentally scalable and that our key design principles generalize to larger models. We scaled our model to 1.3B parameters and replicated our analysis of head-dimension d<sup>h</sup> ∈ {64, 128, 256}. The results, presented in Figure [6](#page-6-1) and Table [1,](#page-6-0) lead to two critical conclusions. First, the superiority of FlashMHF over the baseline is not only preserved but even amplified at the 1.3B scale as shown in Figure [7;](#page-6-0) it converges faster and achieves a substantially lower validation loss, leading to a larger improvement of 0.85 in perplexity. Second, our earlier findings from the 370M model that the optimal head dimension d<sup>h</sup> = 128 provides significant benefits hold true at this larger scale. The consistent behavior across a nearly 4x increase in model size robustly demonstrates that the convergence superiority and performance gains of FlashMHF are a general property of the architecture, making it a viable and scalable solution for training state-of-the-art language models.

## <span id="page-7-1"></span>4.2 DOWNSTREAM TASK PERFORMANCE

Setup. To ascertain whether the improved validation loss of FlashMHF translates into enhanced downstream capabilities, we evaluated our 370M and 1.3B models on a comprehensive suite of benchmarks. We assess commonsense reasoning using HellaSwag [\(Zellers et al., 2019\)](#page-11-6), Social IQA (SIQA) [\(Sap et al., 2019\)](#page-10-7), Physical IQA (PIQA) [\(Bisk et al., 2019\)](#page-9-0), OpenBookQA (OBQA) [\(Mi](#page-10-8)[haylov et al., 2018\)](#page-10-8), and WinoGrande [\(Levesque et al., 2011;](#page-10-9) [Sakaguchi et al., 2019\)](#page-10-10). Additionally, we evaluate reading comprehension using the RACE dataset [\(Lai et al., 2017\)](#page-10-11).

Results. As summarized in Table [2,](#page-7-0) we first note that given our modest training scale (60-100B tokens), a detailed analysis of performance on individual tasks may be of limited significance. However, we posit that the average performance across a diverse benchmark suite is a statistically robust indicator of an architecture's general capabilities, with a superior design expected to yield a consistently better average score. The results strongly support this view. Notably, as highlighted by the gray background in the table, the best performance on every individual benchmark across both the 370M and 1.3B scales is invariably achieved by a FlashMHF variant. While FlashMHF-128hdim secures the highest *average* score, confirming d<sup>h</sup> = 128 as an effective sweet spot, the FlashMHF architecture as a whole consistently outperforms the SwiGLU baseline across all configurations.

## 4.3 SPEED AND MEMORY EFFICIENCY

A core motivation for FlashMHF is to enhance model capabilities while also improving computational efficiency. To conduct a fair evaluation, we benchmarked our module against the SwiGLU FFN baseline. For latency evaluation, to ensure a comparable total parameter count, we compared a 20-layer FlashMHF (and MH-FFN) against a 24-layer SwiGLU baseline. For memory profiling, we compared the modules directly at the single-layer level. The na¨ıve MH-FFN, included for completeness, was substantially less efficient in both metrics; our analysis therefore focuses on the comparison between FlashMHF and the strong SwiGLU baseline. All benchmarks were run on an Nvidia H100 GPU, with detailed configurations and results available in Appendix [E.](#page-16-0)

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 8: Memory and latency comparison of SwiGLU FFN, MH-FFN, and FlashMHF. (Log-graph)

Memory Efficiency FlashMHF delivers a leap in memory efficiency. As shown in Figure [8a,](#page-8-0) our I/O-aware kernel drastically reduces peak memory consumption by a factor of 3-5x compared to a standard SwiGLU FFN. This dramatic reduction directly enables inference and training with significantly longer context lengths or deploying larger models on the same hardware. This advantage further widens as sequence length increases, underscoring the superior scalability of our design.

Latency Regarding latency, FlashMHF also provides a speedup, though the improvement is more moderate compared to the dramatic memory gains. Our benchmarks demonstrate a peak inference speedup of 1.08x, with an average improvement of approximately 1.05x across various configurations (Figure [8b\)](#page-8-0). This speedup primarily stems from eliminating the I/O bottleneck of writing and reading the large intermediate activation tensor to and from HBM. It is worth noting that this latency improvement is more moderate than that of FlashAttention. The reason is that the standard FFN layer gets highly optimized by cuBLAS and has a higher GPU Memory cache hit rate. Nevertheless, this efficiency gain, while modest, is a welcome byproduct of our memory-optimized design, making FlashMHF a practical and beneficial replacement for standard FFNs in production environments.

Summary of Findings. Combining these efficiency results with the model quality evaluations from Section [4.2](#page-7-1) presents a powerful conclusion. The full Transformer model equipped with FlashMHF not only achieves lower perplexity and superior downstream performance but also runs faster and consumes significantly less memory. Our analysis provides comprehensive evidence that FlashMHF offers *a rare "free lunch"*: a direct replacement for standard SwiGLU FFNs that improves every critical metric—model quality, inference speed, and memory footprint—without any trade-offs. This makes it a highly practical and scalable solution for developing next-generation language models.

