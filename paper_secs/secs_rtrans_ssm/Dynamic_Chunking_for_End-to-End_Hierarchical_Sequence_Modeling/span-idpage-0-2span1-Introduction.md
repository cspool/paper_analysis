# <span id="page-0-2"></span>1 Introduction

A broad goal of deep learning is to learn meaningful patterns from raw data, automatically extracting features and building abstractions in an end-to-end fashion. However, fixed-vocabulary tokenization, the process of compressing raw text into predefined chunks through algorithms such as byte-pair encoding (BPE) (Sennrich, Haddow, and Birch [2015;](#page-28-0) Kudo and Richardson [2018\)](#page-26-0), remains a pervasive handcrafted preprocessing step in modern language models (LMs) (Grattafiori et al. [2024;](#page-25-0) Brown et al. [2020\)](#page-24-0). Tokenization comes with a host of well-documented drawbacks, from poor character-level understanding to lack of meaning and interpretability to degraded performance on complex languages and modalities (Petrov et al. [2023;](#page-27-0) Ahia, Kumar, Gonen, Kasai, et al. [2023;](#page-24-1) Belinkov and Bisk [2017;](#page-24-2) Sun et al. [2020;](#page-28-1) Xue et al. [2022;](#page-28-2) Clark, Garrette, et al. [2022\)](#page-24-3). [1](#page-0-0) Replacing the tokenization–LM–detokenization pipeline with a single end-to-end model would also adhere better to the spirit of deep learning, ideally scaling more powerfully with data and parameters (c.f. the bitter lesson) (Sutton [2019;](#page-28-3) Perić [2025\)](#page-27-1). However, tokenization remains an indispensable component of language models and other sequential data for its ability to compress and shorten sequences; as of yet, no end-to-end tokenizer-free model has matched the performance of tokenizer-based language models when matched for computational budget.

A line of recent works has turned to overcoming tokenization in autoregressive sequence models, which requires addressing a series of difficult technical challenges: [2](#page-0-1)

<span id="page-0-0"></span><sup>1</sup>Many other edge cases have been discussed in informal online discourse rather than papers; we defer to Andrej Karpathy's [lectures](https://x.com/karpathy/status/1657949234535211009) and [tweets.](https://x.com/karpathy/status/1759996551378940395)

<span id="page-0-1"></span><sup>2</sup>An extended related work can be found in Appendix [A,](#page-30-0) which is summarized in Table [6.](#page-20-0)

- Direct byte-level language modeling with isotropic architectures<sup>3</sup> can be improved with efficient sequence models such as MambaByte (Wang et al. 2024), but still incur prohibitive computational costs while underperforming tokenized models in compute-matched settings.
- To improve efficiency, hierarchical architectures such as Hourglass Transformer (Nawrot, Tworkowski, et al. 2022) and MegaByte (Yu, Simig, et al. 2023) use small byte-level models to compress raw inputs into subsampled sequences, which are then processed with a more powerful standard language model. However, simple pooling strategies such as compressing every *k* inputs are not data-dependent, and perform poorly on modalities with variable information rates such as language.
- SpaceByte (Slagle 2024) and Byte Latent Transformer (Pagnoni et al. 2024) introduce data-dependent chunking strategies such as delimiter- or entropy-based heuristics. These heuristics, however, rely on auxiliary *external* boundary predictors, and are therefore modality-specific and not fully end-to-end.
- Although jointly trainable boundary predictors are the ideal solution, they require optimizing discrete selection
  operations without supervision, which is fundamentally a challenging problem. Consequently, existing end-to-end
  approaches (Nawrot, Chorowski, et al. 2023) exhibit training instabilities that preclude scaling beyond small models
  or nesting multi-level hierarchies.

Fundamentally, creating a tokenizer-free architecture requires incorporating the data chunking process directly into the model, while overcoming challenges in efficiency, learnability, and stability at scale.

#### Dynamic Chunking: End-to-end Sequence Modeling Without Tokenization

In this work, we introduce an end-to-end **hierarchical network (H-Net)** that compresses raw data through a recursive, data-dependent **dynamic chunking (DC)** process (Figure 1). H-Nets match the efficiency of tokenized pipelines while substantially improving modeling ability, by replacing handcrafted heuristics with content-aware and context-dependent segmentation learned from data.

Hierarchical Processing. The H-Net adopts the hierarchical architecture from prior work (Goel et al. 2022; Nawrot, Tworkowski, et al. 2022; Slagle 2024), resembling an autoregressive U-Net (Ronneberger, Fischer, and Brox 2015): (i) raw data is processed by a small encoder network, (ii) then downsampled and passed through a main network operating on compressed chunks, (iii) and finally upsampled before being passed through a decoder network operating on the original resolution. This modularity creates a natural processing hierarchy where outer stages capture fine-grained patterns while inner stages operate on coarse representations akin to traditional tokens. Crucially, while the main network contains the bulk of parameters and can be any standard architecture designed for operating on tokenized language—such as a Transformer (Vaswani et al. 2017) or state space model (SSM) (Gu and Dao 2024)—we show that the encoder and decoder networks are strongly improved by using SSMs, which have an inductive bias for compression (Gu 2025).

**Dynamic Chunking.** H-Net's core is a novel dynamic chunking (DC) mechanism which interfaces between the main network and the encoder/decoder networks, learning how to segment data while using standard differentiable optimization. DC is composed of two complementary new techniques: (i) a **routing module** which predicts boundaries between adjacent elements through a similarity score (ii) and a **smoothing module** which interpolates representations using the router's outputs, attenuating the effect of uncertain boundaries and significantly improving learnability. By combining these with a new auxiliary loss function that targets desired downsampling ratios, and modern techniques for gradient-based learning of discrete choices (Fedus, Zoph, and Shazeer 2022; Bengio, Léonard, and Courville 2013), DC lets an H-Net learn how to compress data in a fully end-to-end fashion.

**Signal Propagation.** We introduce several architectural and training techniques to improve stability and scalability during end-to-end optimization. These include: (i) carefully placing projections and normalization layers to balance signal propagation between interacting sub-networks, and (ii) adjusting optimization parameters for each layer based on its dimensionality and effective batch size, which changes between stages of the hierarchical structure.

Altogether, H-Net learns segmentation strategies *optimized jointly* with the main backbone, dynamically compressing input vectors based on contextual information into meaningful chunks. H-Net represents the first truly end-to-end, tokenizer-free

<span id="page-1-0"></span><sup>&</sup>lt;sup>3</sup>Non-hierarchical models comprised of repeated blocks, such as the standard Transformer (Vaswani et al. 2017).

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1: **(left)** Architectural overview of H-Net with a two-stage hierarchical design (S = 2). **(right)** Dynamic Chunking (DC). **(bottom-right)** Key components of a chunking layer: (a) a routing module for dynamically drawing chunk boundaries, and (b) a downsampler that selectively retains vectors based on boundary indicators, reducing sequence length while preserving semantically significant positions. **(top-right)** Key components of a dechunking layer: (c) a smoothing module for converting discrete chunks into interpolated representations, and (d) an upsampler that restores compressed vectors to their original resolution based on boundary indicators. Linear in equation (3) and STE in equation (9) are omitted in the illustration for brevity.

language model: with a single stage of dynamic chunking, a *byte-level H-Net* matches the perplexity and downstream performance of a strong *BPE-tokenized Transformer* at sizes exceeding 1B parameters. Empirically, the dynamic chunking module naturally compresses data to a similar resolution as BPE tokenizers (4.5-5 bytes/chunk) and qualitatively learns meaningful boundaries, all without any external supervision or heuristics.

### Hierarchical Chunking: From Data to Abstractions

Beyond addressing tokenization, H-Net improves general sequence modeling across a wide range of settings. Subword tokenization in language models is a special case of *chunking*—the process of building higher-level abstractions from low-level data—and is a central component of intelligence.<sup>4</sup> Crucially, because H-Net is fully end-to-end, **it can be iterated recursively: the main network can itself be an H-Net**. Intuitively, more stages of chunking represent higher order meanings; just as characters can be combined into words, words can be combined into clauses, sentences, and beyond. Iterating the hierarchy should therefore lead to even more efficient use of compute and parameters, and more effective reasoning over compressed representations.

Recursive H-Nets represent a new class of foundation model architectures that not only overcome tokenization, but discover and operate over abstractions learned from raw data, leading to higher-quality models with less pre-processing. Iterating the 1-stage H-Net to 2 hierarchical stages further improves its capabilities and strongly outperforms all baselines, with steeper training curves and better scaling with data. A byte-level 2-stage H-Net overtakes the perplexity of a strong tokenized Transformer after just 30B training bytes, with the gap widening throughout training, and matches the downstream evaluations of the tokenized Transformer of twice its size.

Finally, H-Nets realize the benefits of overcoming tokenization:

- *Robustness*. Without special data mixes, the pretrained H-Net is dramatically more robust to textual perturbations than the token-based Transformer, as evaluated on the noisy HellaSwag suite of benchmarks.
- *Interpretability*. Qualitative visualizations of learned boundaries reveal that H-Net automatically discovers semantically coherent units without explicit supervision, validating that end-to-end learning successfully detects the structural patterns traditionally imposed through handcrafted tokenization.
- Other languages. H-Net's improvements are even more pronounced on languages without obvious segmentation
  cues, including Chinese and code (59.9 → 66.3 on XWinograd-zh compared to tokenized Transformer) and DNA
  language modeling (3.6× improved data efficiency compared to isotropic models).

We publicly release model code<sup>5</sup> and pretrained checkpoints<sup>6</sup>.

### 2 H-Net Architecture

H-Nets are defined as hierarchical U-Net-like networks, but with data-dependent *dynamic subsampling* that is learned end-to-end together with the rest of the model. We first introduce H-Net's hierarchical architecture for multi-level processing, establishing key design principles (Section 2.1). We then present our dynamic chunking mechanism that learns content-aware compression through standard optimization (Section 2.2). Next, we detail architectural and optimization enhancements specifically tailored for hierarchical sequence modeling (Section 2.3). Finally, we explain how H-Net preserves autoregressive properties throughout its hierarchical structure during both training and inference (Section 2.4).

### <span id="page-3-3"></span>2.1 Architectural Overview

#### 2.1.1 Components of H-Net

H-Net employs a hierarchical architecture comprising three primary components – encoder networks ( $\mathcal{E}$ ), main network ( $\mathcal{M}$ ), and decoder networks ( $\mathcal{D}$ ) – where each component is implemented with a stack of sequence mixing layers (e.g., Transformers or state space models). In its simplest form, a single-stage H-Net consists of one encoder network, one main network, and one decoder network. Crucially, the architecture's key characteristic lies in the main network's unique property: it can itself be instantiated as a complete H-Net, enabling recursive construction of multi-level hierarchies.

This recursive design allows H-Net to scale to arbitrary depths. In an S-stage model, we denote components at each stage using superscripts: encoder networks as  $\mathcal{E}^s$  and decoder networks as  $\mathcal{D}^s$  for stages  $0 \le s < S$ , with the main network  $\mathcal{M}$  residing only at the final stage s = S. For example, a two-stage model contains  $\mathcal{E}^0$ ,  $\mathcal{E}^1$ ,  $\mathcal{M}$ ,  $\mathcal{D}^1$ , and  $\mathcal{D}^0$ , as illustrated

<span id="page-3-0"></span><sup>&</sup>lt;sup>4</sup>Chunking is a formal concept from cognitive psychology central to human memory and cognition, and is the inspiration for this work's terminology.

<span id="page-3-1"></span><sup>5</sup>https://github.com/goombalab/hnet

<span id="page-3-2"></span><sup>6</sup>https://huggingface.co/cartesia-ai

in Figure 1-(Left). Throughout this paper, we use superscripts to denote stage indices, though we omit them when all variables within an equation belong to the same stage.

Drawing inspiration from the U-Net architecture (Ronneberger, Fischer, and Brox 2015), H-Net progressively compresses input sequences into fewer vectors with richer semantic embeddings through a chunking layer, processes these representations in the main network, then decompresses the sequence back to its original resolution using a dechunking layer. Unlike traditional U-Net designs, however, H-Net dynamically determines chunking boundaries rather than using fixed-size pooling operations. The overall pipeline can be formalized as:

<span id="page-4-0"></span>
$$\hat{x}^s = \mathcal{E}^s(x^s), \qquad \qquad \hat{z}^S = \mathcal{M}(x^S), \qquad \qquad \hat{z}^s = \mathcal{D}^s(z^s),$$
 (1)

where the chunking layer and the dechunking layer operations are defined as:

<span id="page-4-2"></span>
$$(x^{s+1}, p^s) = \operatorname{Chunk}(\hat{x}^s), \qquad (2) \qquad z^s = \operatorname{Dechunk}(\hat{z}^{s+1}, p^s) + \operatorname{Linear}(\hat{x}^s). \qquad (3)$$

The initial input to the model is  $x^0 \in \mathbb{R}^{L^0 \times D^0}$  where  $L^0$  is the input sequence length and  $D^0$  is the embedding dimension. Intuitively,  $p^s \in [0,1]^{L^s}$  represents the chunking router's confidence that the token should be passed into the main stage. <sup>7</sup> This value is essential for both the chunk (Section 2.2.1) and dechunk operations (Section 2.2.2).

#### 2.1.2 Design Principles

**Encoder and Decoder Networks.** The encoder and decoder networks in H-Net face unique design constraints due to their dual objectives and computational requirements. Each encoder must simultaneously (i) preserve fine-grained information for transmission to its corresponding decoder through residual connections (3), and (ii) compress inputs into chunks of richer representations for the main network. The decoder, in turn, must effectively combine coarse-grained representations from the main network with fine-grained details from the encoder residuals.

Importantly, both encoders and decoders operate on uncompressed sequences, making computational efficiency a significant design constraint that shapes our architectural choices. Recent studies demonstrate that state space models (SSMs) (Gu, Goel, and Ré 2022; Gu and Dao 2024) excel at processing fine-grained data including audio (Goel et al. 2022), DNA sequences (Schiff et al. 2024), and robotic control signals (Lu et al. 2023).

Based on these insights, we employ Mamba-2 layers (Dao and Gu 2024) as the primary building blocks for the encoder and decoder networks. This choice yields two significant benefits: effective handling of fine-grained inputs, and substantially improved efficiency when processing long, uncompressed sequences. Our ablation studies (Section 3.3) confirm that SSM-based encoders/decoders significantly outperform Transformer layers, not just at the byte level but even on coarser inputs, which we attribute to their stronger inductive bias for compression which helps build abstractions (Gu 2025).

**Main Network.** H-Net's computational efficiency stems from strategic parameter allocation. We concentrate the majority of model capacity in the main network, which operates on progressively compressed sequences. After S stages of compression, the main network receives sequences where  $L^S \ll L^0$ , enabling much larger networks within the same computational budget. This design reflects two key principles: (i) compressed sequences allow more parameters and compute per chunk, and (ii) higher-level abstractions benefit from increased processing power.

The main network functions as a standard language model and can employ any sequence mixing architecture. We default to Transformer layers for two reasons: compressed representations align well with Transformers' strengths in processing discrete, semantically-rich tokens, and this choice enables more controlled comparison with traditional BPE-based Transformer baselines in our experiments. However, the modular design also allows straightforward substitution with alternative architectures (e.g., a state space model, hybrid, or H-Net itself) as explored in our ablations.

**Architectural Guidelines.** Compared to standard isotropic models, the H-Net's structure introduces several new dimensions of architectural parameters to balance the parameter/compute allocation to each network. To simplify the search space, we follow a few general guidelines.

<span id="page-4-1"></span><sup>&</sup>lt;sup>7</sup>We also sometimes refer to it as a *probability*—it is interpreted as such in Appendix F—although we do not use it as a formal probability.

- First, we ensure the model width (often referred to as d<sub>model</sub> for isotropic architectures) is monotone in the hierarchy:
   D<sup>0</sup> ≤ D<sup>1</sup> ≤ ··· ≤ D<sup>S</sup>. This allows increasing compute and parameters used in the main network without significantly increasing its depth.
- Second, using efficient and powerful SSM layers in the outer networks allow reducing the number of layers used
  compared to similar prior architectures that only used Transformer layers (Slagle 2024); in this paper, we always
  stick to four layers (or the equivalent of four Mamba layers) in each encoder/decoder network.

To handle the changes in dimensions without an additional linear layer, we adopt the technique used in SpaceByte (Slagle 2024) with the marginal change: to expand dimensions (*i.e.*,  $D^s \to D^{s+1}$ ), we append all vectors with a shared trainable vector of dimension  $D^{s+1} - D^s$ ; to reduce dimensions (*i.e.*,  $D^{s+1} \to D^s$ ), we take the first  $D^s$  dimensions from each vector.

We note that H-Net's performance can likely be improved with more careful tuning of the layer allocation and hyperparameters between sub-networks.

### <span id="page-5-0"></span>2.2 Dynamic Chunking (DC)

H-Net learns chunking boundaries through end-to-end training, allowing it to identify semantically meaningful units adaptively. Furthermore, this dynamic approach enables the model to allocate computational resources efficiently by compressing low-information regions while preserving high-information content at appropriate granularity.

### <span id="page-5-1"></span>2.2.1 Chunking Layer

The chunking layer (Chunk in equation (2)) contains a routing module and downsampler, as illustrated in Figure 1-(bottom-right).

**Routing Module.** In natural data, meaningful boundaries tend to emerge at points of contextual or semantic shift. From this observation, we add an inductive bias by measuring the similarity between adjacent representations: when context changes, consecutive vectors should exhibit lower similarity. The routing module implements this intuition through cosine similarity between adjacent encoder outputs. Given encoder outputs  $\hat{X}$ , it calculates boundary probabilities  $p_t$  and boundary indicators  $b_t$  as follows:

<span id="page-5-3"></span>
$$q_t = W_q \hat{x}_t, \quad k_t = W_k \hat{x}_t, \qquad p_t = \frac{1}{2} \left( 1 - \frac{q_t^\top k_{t-1}}{\|q_t\| \|k_{t-1}\|} \right) \in [0, 1], \quad b_t = \mathbb{1}_{\{p_t \ge 0.5\}}, \tag{4}$$

where  $p_1 = 1.0$  by definition, ensuring the sequence begins with a boundary. This formulation scales cosine similarity into a boundary score or probability: ideally, when consecutive vectors  $\hat{x}_{t-1}$  and  $\hat{x}_t$  span a semantic boundary (e.g., between morphemes, words, or phrases), their projections  $q_t$  and  $k_{t-1}$  diverge in the latent space, yielding low cosine similarity and consequently high boundary probability  $p_t$ .

**Downsampler.** The downsampler compresses encoder outputs  $\hat{x}^s$  into a reduced set of vectors  $x^{s+1}$  using boundary indicators  $\{b_t^s\}_{t=1}^{Ls}$ . Among potential compression strategies – including mean pooling, max pooling, or cross-attention – we adopt direct selection of boundary-marked vectors for its simplicity and effectiveness (see Appendix E.1 for ablations).

As illustrated in Figure 1-(b), this approach follows a straightforward selection rule: vectors where  $b_t = 1$  are retained in the compressed sequence  $x^{s+1}$ , while those where  $b_t = 0$  are discarded. Likewise, the same downsampler applies to boundary probabilities, compressing  $p^s$  into  $P^{s+1}$  for use in a dechunking layer (see Section 2.2.2).

### <span id="page-5-2"></span>2.2.2 Dechunking

The dechunking layer (Dechunk in equation (3)) consists of a smoothing module and upsampler, as illustrated in Figure 1-(top-right).

**Smoothing Module.** The critical challenge in training a dynamic chunking module lies in the discrete nature of chunk boundaries, which impedes gradient flow during backpropagation. We introduce the smoothing module as a technique to address this problem. As illustrated in Figure 1-(c), this component transforms discrete chunking operations into

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Figure 2: Comparison of decompression strategies on the example sequence "...new product!".  $\bullet$  indicates a boundary with high confidence ( $P_t = 1.0$ ) and  $\bullet$  indicates a boundary with low confidence ( $P_t = 0.5$ ). As each letter in the example is unique, we use the letters in subscripts to denote expected semantics of chunks. (a) Optimal chunking with oracle boundaries identifying linguistically meaningful units. (b) Suboptimal chunking without a smoothing module. This creates misalignment during upsampling, causing information from incorrect contexts to propagate. (c) Improved decompression with a smoothing module, where low-confidence chunks are interpolated with weighted combinations of previous chunks, correcting the shaded regions. In panels (b) and (c), we interpret low-confidence boundaries cause the encoder network to embed broader contexts at subsequent positions. Specifically, the vectors at \_ and ! encode new\_ and duct!, respectively (instead of w\_ and ct!).

differentiable computations by creating smooth interpolations between chunks. Concretely, the smoothing module applies an exponential moving average (EMA) with the following definition:

<span id="page-6-6"></span><span id="page-6-5"></span><span id="page-6-0"></span>
$$\bar{z}_t = P_t \hat{z}_t + (1 - P_t) \bar{z}_{t-1}. \tag{5}$$

Our smoothing module performs several roles:

- **Differentiable boundary learning:** It transforms the discrete upsampling operation into a continuous one, enabling effective backpropagation through chunk boundaries during training without requiring stochastic exploration-based approaches (Jang, Gu, and Poole 2017).
- Adaptive error correction: Chunks with high confidence ( $P_t \approx 1.0$ ) maintain discrete boundaries ( $\bar{z}_t \approx z_t$ ), while chunks with low confidence ( $P_t \approx 0.5$ ) are smoothed using information from previous chunks, creating a self-correcting mechanism.
- **Training stability:** By smoothly interpolating between discrete choices based on confidence scores, a smoothing module prevents the model from overfitting to suboptimal chunking patterns early in training.

Figure 2 illustrates this with the example "...new product!". The word "product" can be morphologically decomposed into "pro-" and "-duct". Without the smoothing module (see Figure 2-(b)), suboptimal chunking (e.g., "du" as shown with half-filled circles) creates alignment mismatches that disrupt information flow. With the smoothing module (see Figure 2-(c)), chunks with low confidence are smoothed with previous context, ensuring proper information propagation and enabling the model to learn optimal chunk boundaries through gradient descent.

**Upsampler.** We carefully design the upsampler (see Figure 1-(d)) that decompresses  $\bar{z}^{s+1}$  to match the original resolution of inputs in the previous stage  $z^s$  with the following definition:

<span id="page-6-4"></span><span id="page-6-3"></span>
$$c_t = p_t^{b_t} (1 - p_t)^{1 - b_t} = \begin{cases} p_t & \text{if } b_t = 1, \\ 1 - p_t & \text{otherwise,} \end{cases}$$
 (6) 
$$\tilde{z}_t = \bar{z}_{\sum_{k=1}^t b_k},$$
 (8)

$$STE(c_t) = c_t + stopgradient(1 - c_t),$$
 Upsampler( $\bar{z}, c$ )<sub>t</sub> = STE( $c_t$ ) ·  $\tilde{z}_t$ . (9)

Each component serves a specific purpose in enabling stable end-to-end learning:

<span id="page-6-2"></span><sup>&</sup>lt;sup>8</sup>**pro-** – meaning *forward* or *forth*, **-duct** – from Latin *ducere*, meaning *to lead* or *to bring* 

- Confidence scoring (6): The coefficient c quantifies the routing module's confidence in its boundary decisions. For positions marked as boundaries ( $b_t = 1$ ),  $c_t = p_t$  rewards high boundary probabilities. In contrast, for non-boundary positions ( $b_t = 0$ ),  $c_t = 1 p_t$  penalizes false boundary predictions. This formulation encourages the model to produce boundary probabilities near 1.0 at true boundaries and near 0.0 elsewhere.
- Gradient stabilization (7): The Straight-Through Estimator (STE) (Bengio, Léonard, and Courville 2013) is a well established technique from discrete representation learning (Van Den Oord, Vinyals, et al. 2017; Jang, Gu, and Poole 2017) that rounds confidence scores to 1.0 in the forward pass while maintaining continuous gradients during backpropagation. While H-Net already demonstrates strong performance without STE, incorporating this technique provides an additional performance boost that empirically further stabilizes the optimization dynamics.
- Causal expansion (8): The upsampling operation repeats each compressed vector until the next boundary position, ensuring that each reconstructed position receives information from its most recent chunk. This maintains the sequential flow of information while expanding the compressed representation back to its original length.
- Confidence-weighted decompression (9): Multiplying upsampled vectors by their confidence scores incentivizes the routing module to make confident, accurate decisions. High-confidence boundaries create direct reward signals that encourage the model to sharpen its boundary predictions through gradient feedback.

#### 2.2.3 Ratio Loss

Without explicit regularization, the model may converge to trivial solutions: either retaining nearly all vectors (negating computational benefits) or compressing excessively (losing critical information). Inspired by load balancing mechanisms in Mixture-of-Experts (MoE) models (Fedus, Zoph, and Shazeer 2022), which face similar challenges in maintaining balanced expert utilization, we introduce a ratio loss to guide compression:

<span id="page-7-1"></span>
$$\mathcal{L}_{\text{ratio}} = \frac{N}{N-1} \left( (N-1)FG + (1-F)(1-G) \right), \qquad F = \frac{1}{L} \sum_{t=1}^{L} b_t, \quad G = \frac{1}{L} \sum_{t=1}^{L} p_t, \tag{10}$$

where F represents the fraction of vectors actually selected, G denotes the average boundary probability, and N controls the target compression ratio. Mechanistically, although F is not differentiable, the network can be trained toward targeted compression ratios through G, which provides continuous feedback.

When F = G, the loss attains a minimum of  $\mathcal{L}_{\text{ratio}} = 1$  when  $F = G = \frac{1}{N}$ . Interestingly, the loss can theoretically fall below 1 when  $F \neq G$  (e.g.,  $F = \frac{1}{N} + \epsilon$  and  $G = \frac{1}{N} - \epsilon$ ), which we indeed observe during training. Despite this theoretical possibility, the loss effectively guides the model toward the desired compression ratio in practice. In practice, as our architectural design encourages the routing module to make confident decisions (i.e., boundary probabilities approaching 0 or 1), F naturally converges toward G, and the loss effectively guides the model toward the desired compression ratio.

Combined together with the autoregressive prediction loss (i.e.,  $\mathcal{L} = \mathcal{L}_{AR} + \alpha \sum_{s=0}^{S-1} \mathcal{L}_{ratio}^s$ ), this mechanism preserves content-adaptive compression: the model learns which vectors to retain based on semantic importance rather than following predetermined patterns, distinguishing H-Net from fixed compression schemes. We fixed  $\alpha = 0.03$  in all experiments in this paper as it provides a good balance between prediction accuracy and chunking efficiency; however, in other settings, it may be important to choose this hyperparameter more carefully.

Notationally, we sometimes use  $(N^0, N^1, \dots, N^s)$ -DC to denote the full dynamic chunking mechanism together with its targeted chunking ratios.

### <span id="page-7-0"></span>2.3 Improved Techniques for Hierarchical Sequence Modeling

We introduce several techniques that improve the overall architecture. These may generally be considered techniques to improve *signal propagation* throughout the network, improving stability and learnability.

**Norm Balance.** Modern large language models employ pre-normalization architectures (Radford et al. 2019; Touvron, Lavril, et al. 2023), departing from the post-normalization design of the original Transformer (Vaswani et al. 2017). Following established best practices, these models typically include a final normalization layer after all residual blocks. H-Net adopts this convention through *network normalization*, by placing an RMSNorm (Zhang and Sennrich 2019) at the end of each network component ( $\mathcal{E}^s$ ,  $\mathcal{D}^s$ , and  $\mathcal{M}$ ).

This addition of a normalization layer addresses a critical challenge in hierarchical architectures. Pre-normalization allows residual stream magnitudes to grow unbounded through successive layers, with feature norms increasing monotonically. For H-Net, this poses a particular problem: the architecture leverages residual connections to preserve fine-grained information across stages. Without network normalization, outputs from deeper components (especially the many-layered main network) would dominate the residual signals from earlier encoder networks through imbalanced feature norms, neglecting the fine-grained details that are essential for decompression. The normalization layers restore balance between processed features and residual information, ensuring both contribute meaningfully to the final representation.

Separation of Two Streams. Encoder outputs (ˆ) serve dual purposes in our architecture: passing fine-grained information to corresponding decoders through residual connections, and providing compressed representations as inputs to subsequent stages. This dual functionality creates a design challenge, as these two roles may benefit from different representations. We consider three options to address this: (i) apply a projection to the residual connection only, (ii) apply a projection to the main network inputs only, (iii) and apply a projection to both pathways.

As indicated in equation [\(3\)](#page-4-0), we adopt the first approach – adding a projection (Linear) only to the residual connection. This choice is motivated by the fundamental principle of designing deep learning models (He et al. [2016\)](#page-25-6): maintaining intact gradient flow through the main computational path is crucial for effective training.

Empirically, we found that the third option underperforms despite additional parameters and computations, as the extra projections interfere with gradient propagation. The second option, while preserving residual gradients, disrupts the main network's gradient flow and had worse training dynamics. Our chosen design maintains unimpeded gradients from deeper stages while allowing the residual connection to adapt its contribution through the learned projection. This encourages the model to leverage the main network's computational depth while using residuals in a complementary role.

One additional detail is that this residual connection is initialized close to 0; earlier versions of H-Net found this to be an important detail, but it may be less important when combined with additional techniques such as LR modulation.

Learning Rate Modulation The hierarchical design of H-Net requires careful adjustment of learning rates across stages to ensure balanced training dynamics. Modern theory establishes that neural network hyperparameters should be scaled in predictable ways for optimal trainability (Yang and Hu [2020\)](#page-28-9). Concretely, outer stages, which handle significantly longer input sequences, receive proportionally higher learning rates than inner stages operating on compressed representations. This scaling follows established principles that learning rates are adjusted based on effective batch size and model dimensions. The specific scaling factor we use accounts for both the total number of inputs processed at each stage and the corresponding hidden dimensions (see Appendix [C\)](#page-34-0). [9](#page-8-1) With this modulation, the model achieves more stable training dynamics and improved convergence behavior across the entire hierarchy. In particular, we empirically find that since outer stages directly influence the chunk boundaries that inner stages depend on, the higher learning rates in the outer stages seem to accelerate learning the chunking mechanism.

## <span id="page-8-0"></span>2.4 Autoregressive Training and Inference

Every component of H-Net (i.e., encoder-, decoder-, main- networks, and the dynamic chunking mechanism) is carefully designed to preserve autoregressive properties essential for language modeling.

Training. During training, H-Net employs standard causal masking across all sequence mixing layers. DC maintains causality by computing boundary probabilities based only on current and previous representations. Specifically, the boundary probability depends on and from the current and previous positions (equation [\(4\)](#page-5-3)), ensuring no information leakage from future tokens. The smoothing module similarly maintains causality through its recursive formulation (equation [\(5\)](#page-6-6)), where each output depends only on past compressed representations.

Inference. For inference, H-Net generates raw bytes (or whatever the outermost modality is) autoregressively with a modified procedure to handle its hierarchical structure.

Generation with a prompt proceeds as follows:

<span id="page-8-1"></span><sup>9</sup>We later realized that SpaceByte also followed muP LR scaling (Yang and Hu [2020\)](#page-28-9) to account for model dimension (Slagle [2024,](#page-28-5) Appendix B.2) (but did not account for batch size scaling, as we do). Our reimplementation did not do this and therefore is not fully faithful to the original SpaceByte.

- 1. Initial processing: During prefill, we generate chunks via the encoders (as in training). For each component (i.e. the isotropic components, and the routing module and dechunking layer), we generate a state. Isotropic state (e.g. KV cache for Transformer layers, SSM state for Mamba-2 layers) is generated as usual.
- 2. DC state and DC step: As noted above, the DC modules have recursive formulations that maintain causality at train-time. These recursive formulations become autoregressive formulations at inference time.
  - (a) Routing Module: In order to compute , we need −<sup>1</sup> (see equation [\(4\)](#page-5-3)), so our state consists of the key value of the most recent token processed.
  - (b) Dechunking Layer: In order to compute ˜ , we need and ˜−1. Thus, the dechunking layer state should consist of the last ˜ value.
- 3. Token Generation:[10](#page-9-0) To perform a model step, we do the following for a 1-stage hierarchy:
  - (a) Pass the token through the encoder network,
  - (b) Step the routing module to determine whether the token needs to be processed by the main network,
  - (c) Step the main network if necessary, in which case we also need to step the dechunking layer.
  - (d) Use the result of the dechunking layer to step the decoder network.

A consequence of this inference formulation is that, at inference time, H-Net decides individually for each token how much compute to use when processing it. Therefore, H-Net can allocate more or less compute to different tokens as it deems necessary. A particular connection is that inference resembles speculative decoding (Leviathan, Kalman, and Matias [2023;](#page-26-3) Chen et al. [2023\)](#page-24-6), which also involves a small network (the draft model) stepping on every token, and a larger network (the verification model) only stepping on contiguous chunks of every few tokens.

